"""v0.2: N-1 security, restoration by a parallel circuit or an open tie, reliability in the plan choice, AC check.

Needs results/cost_table_full_year.npz of run_full_year.py. The only new operating cost table is for the plan with a
parallel cable on every line (about 25 minutes on six cores); it is cached in results/checkpoints/.
"""

import json
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from feederrisk import decision as dc
from feederrisk import postfault as pf
from feederrisk.accheck import build_ac_net
from feederrisk.config import DER_LEVELS, LOAD_LEVELS, Assumptions
from feederrisk.daylp import DayModel, Plan
from feederrisk.study import CENTRAL, build_plans, day_arrays, load_inputs, scenario_probs

from run_study import SOURCES, capex_vector

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CHECK = RES / "checkpoints"
WORKERS = 6
DER = list(DER_LEVELS.values())
LOAD = list(LOAD_LEVELS.values())


def _all_lines_chunk(args):
    inp, plan, a, w, p = args
    model = DayModel(inp.feeder, plan, a)
    out = np.zeros((len(DER), len(LOAD)))
    for d, l in product(range(len(DER)), range(len(LOAD))):
        out[d, l] = sum(model.solve(*day_arrays(inp, w, p, d, day, l)).cost for day in range(365))
    return w, p, out


def all_lines_operation(inp, a) -> np.ndarray:
    """Operating cost of the plan with a parallel cable on every line, shape (weather, price, DER, load)."""
    path = CHECK / "all_lines.npz"
    if path.exists():
        return np.load(path)["op"]
    plan = Plan("all_lines", -1, 0.0, tuple(range(inp.feeder.n)))
    shape = (inp.wind_cf.shape[0], inp.price.shape[0])
    jobs = [(inp, plan, a, w, p) for w in range(shape[0]) for p in range(shape[1])]
    op = np.zeros((*shape, len(DER), len(LOAD)))
    with Pool(WORKERS) as pool:
        for w, p, o in pool.imap_unordered(_all_lines_chunk, jobs):
            op[w, p] = o
            print("all_lines chunk", w, p, flush=True)
    CHECK.mkdir(exist_ok=True)
    np.savez_compressed(path, op=op)
    return op


def reliability_tensor(inp, a, rel, structures):
    """ENS and generation lost (MWh per year) by structure and scenario (weather, DER, load)."""
    f = inp.feeder
    ens = {s: np.zeros((inp.wind_cf.shape[0], len(DER), len(LOAD))) for s in structures}
    lost = {s: np.zeros_like(ens[s]) for s in structures}
    for w, d, l in product(range(inp.wind_cf.shape[0]), range(len(DER)), range(len(LOAD))):
        load, gen = pf.scenario_arrays(f, inp.load_h, inp.wind_cf[w], inp.solar_cf[w], DER[d], LOAD[l])
        for s in structures:
            r = pf.reliability(f, s[0], s[1], load, gen, a, rel)
            ens[s][w, d, l] = r["ens_mwh"]
            lost[s][w, d, l] = r["generation_lost_mwh"]
    return ens, lost


def _ac_job(args):
    """AC power flow on the post-fault network, on a sample of hours, against the linear verdict."""
    import pandapower as pp
    inp, doubled, tie_on, line, scen, rel, a, step = args
    f = inp.feeder
    w, d, l = scen
    load, gen = pf.scenario_arrays(f, inp.load_h, inp.wind_cf[w], inp.solar_cf[w], DER[d], LOAD[l])
    q = load * f.load_q_share
    tie = pf.tie_buses(f) if tie_on else None
    tree = pf.post_fault_tree(f, doubled, line, tie, rel.tie_km)
    if not tree.reach[pf.subtree(f, line)].all():
        return None
    # linear verdict and the generation share at which it holds
    best = np.full(load.shape[0], -1.0)
    for alpha in pf.ALPHAS:
        ok = pf.feasible(tree, load - alpha * gen, q, a)
        best = np.where((best < 0) & ok, alpha, best)
    hours = np.arange(0, load.shape[0], step)
    net = build_ac_net(f.reinforced(doubled), a)
    kv3 = a.v_nominal_kv * np.sqrt(3)
    if tie_on:
        per_km = lambda v: v[3] / f.length_km[3]
        pp.create_line_from_parameters(net, tie[0] + 1, tie[1] + 1, length_km=1.0, r_ohm_per_km=per_km(f.r_ohm) * rel.tie_km,
                                       x_ohm_per_km=per_km(f.x_ohm) * rel.tie_km, c_nf_per_km=0.0,
                                       max_i_ka=f.s_max_mva[3] / kv3)
    if line in doubled:
        net.line.at[line, "r_ohm_per_km"] = f.r_ohm[line]
        net.line.at[line, "x_ohm_per_km"] = f.x_ohm[line]
        net.line.at[line, "max_i_ka"] = f.s_max_mva[line] / kv3
    else:
        net.line.at[line, "in_service"] = False
    agree = false_feasible = false_infeasible = 0
    v_err = loading = viol_v = 0.0
    v_lin_all, _ = pf.linear_flow(tree, load - np.maximum(best, 0.0)[:, None] * gen, q, a)
    for h in hours:
        alpha = max(best[h], 0.0)
        net.load["p_mw"] = load[h] - alpha * gen[h]
        net.load["q_mvar"] = q[h]
        pp.runpp(net, numba=False)
        vm = net.res_bus.vm_pu.to_numpy()[1:]
        r = tree.reach
        ok_ac = bool(((vm[r] >= a.v_min_pu - 1e-4) & (vm[r] <= a.v_max_pu + 1e-4)).all()
                     and (net.res_line.loading_percent[net.line.in_service].max() <= 100.0 + 1e-6))
        lin_ok = best[h] >= 0
        agree += int(ok_ac == lin_ok)
        false_feasible += int(lin_ok and not ok_ac)
        false_infeasible += int((not lin_ok) and ok_ac)
        v_err = max(v_err, float(np.abs(vm[r] - v_lin_all[h][r]).max()))
        loading = max(loading, float(net.res_line.loading_percent[net.line.in_service].max()))
        viol_v = max(viol_v, float(max(a.v_min_pu - vm[r].min(), vm[r].max() - a.v_max_pu, 0.0)))
    return {"doubled": len(doubled), "tie": tie_on, "line": line, "scenario": str(scen), "hours": len(hours),
            "agree": agree, "false_feasible": false_feasible, "false_infeasible": false_infeasible,
            "v_err_max": v_err, "loading_max_pct": loading, "violation_v_max": viol_v}


def main() -> None:
    inp = load_inputs(ROOT / "data" / "inputs.npz")
    f, a, rel = inp.feeder, Assumptions(), pf.Reliability()
    plans = build_plans(f)
    probs = scenario_probs(inp.wind_cf.shape[0], inp.price.shape[0])
    base = np.load(RES / "cost_table_full_year.npz")
    assert [str(x) for x in base["plans"]] == [p.name for p in plans]
    op_all = all_lines_operation(inp, a)
    trunk = next(p.lines for p in plans if p.name == "trunk")
    wind_lines = next(p.lines for p in plans if p.name == "wind_lines")
    every = tuple(range(f.n))

    ext = [(p, False) for p in plans] + [(p, True) for p in plans] + [(Plan("all_lines", -1, 0.0, every), False)]
    name = lambda p, t: p.name + ("+tie" if t else "")
    names = [name(p, t) for p, t in ext]
    op = np.concatenate([base["op"], base["op"], op_all[None]], axis=0)
    structures = sorted({(tuple(p.lines), t) for p, t in ext})
    ens, lost = reliability_tensor(inp, a, rel, structures)
    tie_capex = rel.tie_km * a.cable_annual_eur_per_km()
    capex = np.array([p.capex_annual(a, f.length_km) + (tie_capex if t else 0.0) for p, t in ext])
    wdl = lambda arr: float((arr * probs[0][:, None, None] * probs[2][None, :, None] * probs[3][None, None, :]).sum())

    def total_cost(voll, fault_scale, tie_cost=tie_capex):
        rel_cost = np.stack([pf.cost_from_parts(ens[(tuple(p.lines), t)], lost[(tuple(p.lines), t)], a, voll, fault_scale)
                             for p, t in ext])                       # (plans, weather, DER, load)
        cap = np.array([p.capex_annual(a, f.length_km) + (tie_cost if t else 0.0) for p, t in ext])
        return op + cap.reshape(-1, 1, 1, 1, 1) + rel_cost[:, :, None, :, :]

    Q0 = op + capex.reshape(-1, 1, 1, 1, 1)                          # without reliability
    Q = total_cost(rel.voll_eur_mwh, 1.0)
    e0, e = dc.expected(Q0, probs), dc.expected(Q, probs)
    rel_exp = e - e0
    table = pd.DataFrame({
        "plan": names, "cable_km": [float(f.length_km[list(p.lines)].sum()) if p.lines else 0.0 for p, _ in ext],
        "tie": [t for _, t in ext], "battery_mwh": [p.energy_mwh for p, _ in ext],
        "annual_capex_keur": capex / 1e3, "expected_cost_without_reliability_keur": e0 / 1e3,
        "expected_reliability_cost_keur": rel_exp / 1e3, "expected_cost_keur": e / 1e3,
        "cvar90_keur": dc.cvars(Q, probs, 0.9) / 1e3,
        "ens_mwh": [wdl(ens[(tuple(p.lines), t)]) for p, t in ext]}).sort_values("expected_cost_keur").round(2)
    table.to_csv(RES / "n1_plans.csv", index=False)

    out = {"assumptions": {"fault_per_km_y": rel.fault_per_km_y, "repair_h": rel.repair_h, "switching_h": rel.switching_h,
                           "tie_km": rel.tie_km, "tie_annual_capex_keur": tie_capex / 1e3,
                           "voll_eur_kwh": rel.voll_eur_mwh / 1e3},
           "tie_buses": [int(b) for b in pf.tie_buses(f)]}
    i0, i1 = int(e0.argmin()), int(e.argmin())
    out["best_without_reliability"] = {"plan": names[i0], "expected_keur": float(e0[i0] / 1e3)}
    out["best_with_reliability"] = {"plan": names[i1], "expected_keur": float(e[i1] / 1e3)}
    for src_i, src in enumerate(SOURCES):
        out[f"evpi_{src}_without"] = dc.evpi(Q0, probs, (src_i,))
        out[f"evpi_{src}_with"] = dc.evpi(Q, probs, (src_i,))
    out["evpi_total_without"], out["evpi_total_with"] = dc.evpi(Q0, probs), dc.evpi(Q, probs)
    out["central_plan_with"] = names[dc.central_plan(Q, CENTRAL)]

    # Reliability of the main structures by scenario, and line by line at the central scenario
    rows, lines = [], []
    for label, s in (("nothing", ((), False)), ("tie", ((), True)), ("trunk", (trunk, False)), ("trunk+tie", (trunk, True)),
                     ("wind_lines", (wind_lines, False)), ("all_lines", (every, False))):
        for w, d, l in product(range(3), range(3), range(3)):
            c = pf.cost_from_parts(ens[s][w, d, l], lost[s][w, d, l], a, rel.voll_eur_mwh)
            rows.append({"structure": label, "weather": w, "der": DER[d], "load": LOAD[l], "ens_mwh_per_year": ens[s][w, d, l],
                         "generation_lost_mwh": lost[s][w, d, l], "reliability_cost_keur": c / 1e3})
        load_h, gen_h = pf.scenario_arrays(f, inp.load_h, inp.wind_cf[1], inp.solar_cf[1], DER[1], LOAD[1])
        for pl in pf.reliability(f, s[0], s[1], load_h, gen_h, a, rel)["per_line"]:
            lines.append({"structure": label, **pl})
    pd.DataFrame(rows).round(4).to_csv(RES / "n1_reliability.csv", index=False)
    pd.DataFrame(lines).round(4).to_csv(RES / "n1_lines.csv", index=False)

    # N-1 compliance: every line restored (within the switching time at most) in at least 99% of the hours of every scenario
    comp = {}
    for label, s in (("nothing", ((), False)), ("tie", ((), True)), ("trunk", (trunk, False)), ("trunk+tie", (trunk, True)),
                     ("all_lines", (every, False))):
        worst = 1.0
        for w, d, l in product(range(3), range(3), range(3)):
            ld, gn = pf.scenario_arrays(f, inp.load_h, inp.wind_cf[w], inp.solar_cf[w], DER[d], LOAD[l])
            for line in range(f.n):
                o = pf.fault_outcome(f, s[0], s[1], line, ld, gn, a, rel)
                worst = min(worst, float(o["restored"].mean()))
        comp[label] = worst
    out["worst_restored_hours_share"] = comp
    ia = names.index("all_lines")
    out["price_of_full_n1"] = {"all_lines_expected_keur": float(e[ia] / 1e3), "best_expected_keur": float(e[i1] / 1e3),
                               "difference_keur": float((e[ia] - e[i1]) / 1e3)}

    # Sensitivities: VOLL, fault rate, tie cost
    sens = []
    for voll in (3_000, 10_000, 30_000, 100_000):
        for fs in (0.4, 1.0, 2.0):
            for tc in (0.5 * tie_capex, tie_capex):
                Qs = total_cost(voll, fs, tc)
                es = dc.expected(Qs, probs)
                j = int(es.argmin())
                sens.append({"voll_eur_kwh": voll / 1e3, "fault_rate_x": fs, "tie_cost_x": tc / tie_capex, "best_plan": names[j],
                             "expected_keur": es[j] / 1e3, "tie_in_best": names[j].endswith("+tie"),
                             "evpi_load_keur": dc.evpi(Qs, probs, (3,)) / 1e3, "evpi_der_keur": dc.evpi(Qs, probs, (2,)) / 1e3})
    pd.DataFrame(sens).round(2).to_csv(RES / "n1_sensitivity.csv", index=False)
    # Break-even VOLL of the tie on top of the best plan without it
    be = None
    for voll in np.arange(1_000, 300_001, 1_000):
        es = dc.expected(total_cost(float(voll), 1.0), probs)
        if names[int(es.argmin())].endswith("+tie"):
            be = float(voll)
            break
    out["breakeven_voll_for_a_tie_eur_kwh"] = None if be is None else be / 1e3

    # AC check of the post-fault states
    scenarios = [(1, 1, 1), (1, 2, 2)]
    jobs = [(inp, d_, t, line, sc, rel, a, 17) for (d_, t) in ((trunk, False), (every, False), ((), True), (trunk, True))
            for line in range(f.n) for sc in scenarios]
    with Pool(WORKERS) as pool:
        res = [r for r in pool.map(_ac_job, jobs, chunksize=1) if r is not None]
    ac = pd.DataFrame(res)
    ac.to_csv(RES / "n1_ac_check.csv", index=False, float_format="%.5g")
    out["ac_check"] = {"cases": int(len(ac)), "hours": int(ac["hours"].sum()), "agree_share": float(ac["agree"].sum() / ac["hours"].sum()),
                       "false_feasible_hours": int(ac["false_feasible"].sum()),
                       "false_infeasible_hours": int(ac["false_infeasible"].sum()),
                       "v_err_max_pu": float(ac["v_err_max"].max()), "violation_v_max_pu": float(ac["violation_v_max"].max()),
                       "loading_max_pct": float(ac["loading_max_pct"].max())}
    (RES / "summary_n1.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    main()
