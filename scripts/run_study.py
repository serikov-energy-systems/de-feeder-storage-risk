"""Representative-day study, sensitivities and checks, written to results/. About 3 hours on six cores."""

import ctypes
import json
import sys
import time
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

from feederrisk import decision as dc
from feederrisk.accheck import build_ac_net, check_day
from feederrisk.config import Assumptions, DER_LEVELS, PRICE_YEARS
from feederrisk.daylp import DayModel
from feederrisk.study import (CENTRAL, SOURCES, Inputs, blind, build_plans, cost_table, day_arrays,
                              full_year_costs, load_inputs, representative_days, scenario_probs)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
WORKERS = 6


def cached_table(tag, inp, plans, a):
    """Each finished table goes to disk, so a stopped run resumes where it ended."""
    path = RES / "checkpoints" / f"{tag}.npz"
    if path.exists():
        d = np.load(path)
        if d["plans"].tolist() == [p.name for p in plans]:
            print("loaded", tag, flush=True)
            rep = {(w, q): representative_days(inp, w, q) for w in range(inp.wind_cf.shape[0]) for q in range(inp.price.shape[0])}
            return d["op"], d["parts"], rep
    start = time.time()
    op, parts, rep = cost_table(inp, plans, a, workers=WORKERS)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, op=op, parts=parts, plans=np.array([p.name for p in plans]))
    tmp.replace(path)
    print("done", tag, round((time.time() - start) / 60, 1), "min", flush=True)
    return op, parts, rep


def capex_vector(plans, a, f):
    return np.array([p.capex_annual(a, f.length_km) for p in plans])


def with_capex(op, plans, a, f):
    return op + capex_vector(plans, a, f).reshape(-1, 1, 1, 1, 1)


def summary(Q, plans, probs):
    e = dc.expected(Q, probs)
    i = int(e.argmin())
    c = dc.central_plan(Q, CENTRAL)
    return {
        "best_plan": plans[i].name,
        "expected_cost": float(e[i]),
        "central_plan": plans[c].name,
        "vss": dc.vss(Q, probs, CENTRAL),
        "evpi_total": dc.evpi(Q, probs),
        **{f"evpi_{s}": dc.evpi(Q, probs, (k,)) for k, s in enumerate(SOURCES)},
    }


def main() -> None:
    RES.mkdir(exist_ok=True)
    if sys.platform == "win32":
        # Below normal priority, inherited by the workers, keeps the computer usable.
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    inp: Inputs = load_inputs(ROOT / "data" / "inputs.npz")
    f = inp.feeder
    a = Assumptions()
    plans = build_plans(f)
    probs = scenario_probs(inp.wind_cf.shape[0], inp.price.shape[0])
    print(len(plans), "plans")

    op, parts, rep = cached_table("base", inp, plans, a)
    np.savez_compressed(RES / "cost_table.npz", op=op, parts=parts, plans=np.array([p.name for p in plans]))
    Q = with_capex(op, plans, a, f)

    e = dc.expected(Q, probs)
    risk_table = pd.DataFrame({
        "plan": [p.name for p in plans],
        "battery_mwh": [p.energy_mwh for p in plans],
        "cable_km": [float(f.length_km[list(p.lines)].sum()) if p.lines else 0.0 for p in plans],
        "annual_capex_keur": capex_vector(plans, a, f) / 1e3,
        "expected_cost_keur": e / 1e3,
        "cvar90_keur": dc.cvars(Q, probs, 0.9) / 1e3,
        "worst_keur": Q.reshape(len(plans), -1).max(axis=1) / 1e3,
        "max_regret_keur": dc.regret(Q).reshape(len(plans), -1).max(axis=1) / 1e3,
        "arbitrage_keur": dc.expected(parts[..., 0], probs) / 1e3,
        "curtailed_mwh": dc.expected(parts[..., 1], probs),
        "shed_mwh": dc.expected(parts[..., 2], probs),
        "overlap_mwh": dc.expected(parts[..., 3], probs),
    }).sort_values("expected_cost_keur")
    risk_table.round(2).to_csv(RES / "plans.csv", index=False)

    base = summary(Q, plans, probs)
    base["n_plans"] = len(plans)
    base["n_scenarios"] = int(np.prod(Q.shape[1:]))
    betas = {}
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        obj = dc.risk_objective(Q, probs, beta)
        i = int(obj.argmin())
        betas[str(beta)] = {"plan": plans[i].name, "expected": float(e[i]), "cvar90": float(dc.cvars(Q, probs, 0.9)[i])}
    base["risk_sweep"] = betas
    base["evsi_der"] = {str(acc): dc.evsi(Q, probs, 2, acc) for acc in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    base["evsi_price"] = {str(acc): dc.evsi(Q, probs, 1, acc) for acc in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}

    # Best plan and information value in each DER state.
    per_der = {}
    cond = dc._marginal(Q, probs, (2,))
    for d, name in enumerate(DER_LEVELS):
        i = int(cond[:, d].argmin())
        per_der[name] = {"plan": plans[i].name, "cost": float(cond[i, d])}
    base["best_by_der"] = per_der
    cond = dc._marginal(Q, probs, (1,))
    base["best_by_price"] = {str(y): plans[int(cond[:, k].argmin())].name for k, y in enumerate(PRICE_YEARS)}

    # Battery capital cost and cable cost act only on the first stage, so they sweep without new runs.
    sweep = []
    for kwh in (100, 150, 200, 250, 300, 350, 400, 500):
        for km in (60_000, 120_000, 240_000):
            b = replace(a, battery_capex_eur_kwh=float(kwh), cable_capex_eur_km=float(km))
            sweep.append({"battery_eur_kwh": kwh, "cable_eur_km": km, **summary(with_capex(op, plans, b, f), plans, probs)})
    pd.DataFrame(sweep).round(1).to_csv(RES / "capex_sweep.csv", index=False)

    # Sensitivities run on the full plan set.
    keep = [p.name for p in plans]
    by_name = {p.name: i for i, p in enumerate(plans)}
    sub = [by_name[n] for n in keep]
    sub_plans = [plans[i] for i in sub]
    base["sensitivity_plans"] = keep

    def checkpoint() -> None:
        (RES / "summary.json").write_text(json.dumps(base, indent=2))

    checkpoint()

    # The operating value of curtailment changes the second stage, so it needs new runs.
    curtail = {}
    for val in (40.0, 120.0):
        b = replace(a, curtail_eur_mwh=val)
        op2, _, _ = cached_table(f"curtail_{val:g}", inp, sub_plans, b)
        curtail[str(val)] = summary(with_capex(op2, sub_plans, b, f), sub_plans, probs)
        base["curtail_sensitivity"] = curtail
        checkpoint()

    # The substation setpoint decides the voltage headroom, and voltage is the limit that binds.
    voltage = {}
    for val in (1.00, 1.01, 1.03):
        b = replace(a, v_slack_pu=val)
        op2, parts2, _ = cached_table(f"slack_{val:g}", inp, sub_plans, b)
        voltage[str(val)] = summary(with_capex(op2, sub_plans, b, f), sub_plans, probs)
        voltage[str(val)]["curtailed_mwh_no_action"] = float(dc.expected(parts2[..., 1], probs)[0])
        base["slack_voltage_sensitivity"] = voltage
        checkpoint()

    # A plan chosen on a copper plate, then judged on the real feeder.
    a_blind, f_blind = blind(a, f)
    inp_blind = Inputs(f_blind, inp.load_h, inp.wind_cf, inp.solar_cf, inp.price, inp.weather_years, inp.extras)
    op_b, _, _ = cached_table("copper_plate", inp_blind, sub_plans, a_blind)
    op_real = op[sub]
    e_real = dc.expected(op_real + capex_vector(sub_plans, a, f).reshape(-1, 1, 1, 1, 1), probs)
    e_b = dc.expected(op_b + capex_vector(sub_plans, a, f).reshape(-1, 1, 1, 1, 1), probs)
    i_b = int(e_b.argmin())
    base["copper_plate"] = {
        "chosen": sub_plans[i_b].name,
        "believed_cost": float(e_b[i_b]),
        "real_cost": float(e_real[i_b]),
        "regret_against_best": float(e_real[i_b] - e_real.min()),
    }
    # At a low battery cost the copper plate does choose a battery, and only the real feeder tells the site.
    cheap = replace(a, battery_capex_eur_kwh=150.0)
    capex_cheap = capex_vector(sub_plans, cheap, f).reshape(-1, 1, 1, 1, 1)
    e_cheap_blind = dc.expected(op_b + capex_cheap, probs)
    e_cheap_real = dc.expected(op_real + capex_cheap, probs)
    pick = sub_plans[int(e_cheap_blind.argmin())]
    sites = [i for i, p in enumerate(sub_plans) if p.energy_mwh == pick.energy_mwh and not p.lines and p.bus >= 0]
    base["copper_plate_cheap_battery"] = {
        "battery_eur_kwh": 150,
        "chosen": pick.name,
        "chosen_size_mwh": pick.energy_mwh,
        "believed_cost": float(e_cheap_blind.min()),
        "real_cost_by_site": {sub_plans[i].name: float(e_cheap_real[i]) for i in sites},
        "real_best": sub_plans[int(e_cheap_real.argmin())].name,
        "real_best_cost": float(e_cheap_real.min()),
    }
    checkpoint()

    # Representative days against the full year.
    best = int(e.argmin())
    check_plans = list(dict.fromkeys([0, best, dc.central_plan(Q, CENTRAL), by_name["8MWh@far_wind"], by_name["8MWh@head"]]))
    scenarios = [(1, 1, 0, 1), (1, 1, 1, 1), (1, 1, 2, 1), (0, 0, 2, 2), (2, 2, 2, 0)]
    full = full_year_costs(inp, [plans[i] for i in check_plans], a, scenarios, workers=WORKERS)
    rows = []
    for ip, i in enumerate(check_plans):
        for js, s in enumerate(scenarios):
            rows.append({"plan": plans[i].name, "scenario": str(s), "full_year_keur": full[ip, js] / 1e3,
                         "representative_keur": op[(i, *s)] / 1e3})
    pd.DataFrame(rows).round(2).to_csv(RES / "day_selection_check.csv", index=False)

    # AC power flow on the LP dispatch.
    ac = []
    for i in check_plans:
        plan = plans[i]
        model = DayModel(f, plan, a)
        net = build_ac_net(f.reinforced(plan.lines), a)
        f_real = f.reinforced(plan.lines)
        for s in [(1, 1, 1, 1), (1, 1, 2, 1), (2, 0, 2, 0)]:
            days, _ = rep[(s[0], s[1])]
            stats = [check_day(net, f_real, plan, a, *day_arrays(inp, s[0], s[1], s[2], int(day), s[3]), model) for day in days]
            cat = {k: np.concatenate([x[k] for x in stats]) for k in stats[0]}
            ac.append({
                "plan": plan.name, "scenario": str(s), "hours": len(cat["v_err"]),
                "v_err_max": cat["v_err"].max(), "v_max": cat["v_max"].max(), "v_min": cat["v_min"].min(),
                "hours_over_v": int((cat["v_max"] > a.v_max_pu + 1e-4).sum()),
                "hours_under_v": int((cat["v_min"] < a.v_min_pu - 1e-4).sum()),
                "loading_max_pct": cat["loading"].max(),
                "hours_over_100pct": int((cat["loading"] > 100.0 + 1e-3).sum()),
                "loss_share_max": float((cat["loss_mw"] / np.maximum(cat["feed_mw"], 0.05)).max()),
            })
    pd.DataFrame(ac).round(4).to_csv(RES / "ac_check.csv", index=False)

    base["inputs"] = {
        "weather_years": inp.weather_years.tolist(),
        "weather_annual_cf": inp.extras["weather_annual_cf"].round(4).tolist(),
        "feeder_km": float(f.length_km.sum()),
        "buses": f.n,
        "wind_mw": float(f.wind_mw.sum()),
        "pv_mw": float(f.pv_mw.sum()),
        "peak_load_mw": float(inp.load_h.sum(axis=1).max()),
        "rep_days_per_pair": int(np.mean([len(v[0]) for v in rep.values()])),
    }
    (RES / "summary.json").write_text(json.dumps(base, indent=2))
    print(json.dumps(base, indent=2))
    print(risk_table.head(12).round(1).to_string())


if __name__ == "__main__":
    main()
