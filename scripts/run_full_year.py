"""Base case on all 365 days: every plan in every scenario, with no representative days.

The sensitivities in run_study.py use representative days. The check there showed that those days understate
battery arbitrage and high-DER curtailment, so the headline numbers come from this script. Each plan goes to
disk when it ends, so a stopped run resumes. About 100 minutes on six cores.
"""

import ctypes
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from feederrisk import decision as dc
from feederrisk.accheck import build_ac_net, check_day
from feederrisk.config import DER_LEVELS, LOAD_LEVELS, PRICE_YEARS, Assumptions
from feederrisk.daylp import DayModel
from feederrisk.study import CENTRAL, _plan_table, build_plans, day_arrays, load_inputs, scenario_probs

from run_study import capex_vector, summary, with_capex

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CHECK = RES / "checkpoints" / "year"
WORKERS = 6


def one_plan(args):
    i, job = args
    op, parts = _plan_table(job)
    tmp = CHECK / f"{i:02d}.tmp.npz"
    np.savez_compressed(tmp, op=op, parts=parts, name=job[1].name)
    tmp.replace(CHECK / f"{i:02d}.npz")
    print("plan", i, job[1].name, flush=True)
    return i


def main() -> None:
    CHECK.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    inp = load_inputs(ROOT / "data" / "inputs.npz")
    f = inp.feeder
    a = Assumptions()
    plans = build_plans(f)
    probs = scenario_probs(inp.wind_cf.shape[0], inp.price.shape[0])
    shape = (inp.wind_cf.shape[0], inp.price.shape[0], len(DER_LEVELS), len(LOAD_LEVELS))

    every_day = (np.arange(365), np.ones(365))
    rep = {(w, p): every_day for w in range(shape[0]) for p in range(shape[1])}
    jobs = [(i, (inp, plan, a, rep, shape)) for i, plan in enumerate(plans) if not (CHECK / f"{i:02d}.npz").exists()]
    print(len(plans), "plans,", len(jobs), "to run", flush=True)
    if jobs:
        with Pool(WORKERS) as pool:
            for _ in pool.imap_unordered(one_plan, jobs, chunksize=1):
                pass

    loaded = [np.load(CHECK / f"{i:02d}.npz") for i in range(len(plans))]
    assert [str(d["name"]) for d in loaded] == [p.name for p in plans]
    op = np.stack([d["op"] for d in loaded])
    parts = np.stack([d["parts"] for d in loaded])
    np.savez_compressed(RES / "cost_table_full_year.npz", op=op, parts=parts, plans=np.array([p.name for p in plans]))

    Q = with_capex(op, plans, a, f)
    e = dc.expected(Q, probs)
    pd.DataFrame({
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
    }).sort_values("expected_cost_keur").round(2).to_csv(RES / "plans_full_year.csv", index=False)

    out = summary(Q, plans, probs)
    out["evsi_der"] = {str(x): dc.evsi(Q, probs, 2, x) for x in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    out["risk_sweep"] = {}
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = int(dc.risk_objective(Q, probs, beta).argmin())
        out["risk_sweep"][str(beta)] = {"plan": plans[i].name, "expected": float(e[i]), "cvar90": float(dc.cvars(Q, probs, 0.9)[i])}
    cond = dc._marginal(Q, probs, (2,))
    out["best_by_der"] = {n: {"plan": plans[int(cond[:, d].argmin())].name, "cost": float(cond[:, d].min())} for d, n in enumerate(DER_LEVELS)}
    cond = dc._marginal(Q, probs, (1,))
    out["best_by_price"] = {str(y): plans[int(cond[:, k].argmin())].name for k, y in enumerate(PRICE_YEARS)}

    # Exact break-even between battery and no battery, on a fine grid of capital cost.
    flips = {}
    for km in (60_000, 120_000, 240_000):
        rows = []
        for kwh in range(100, 501, 5):
            b = replace(a, battery_capex_eur_kwh=float(kwh), cable_capex_eur_km=float(km))
            ee = dc.expected(with_capex(op, plans, b, f), probs)
            rows.append((kwh, plans[int(ee.argmin())]))
        flips[str(km)] = next((k for k, p in rows if p.energy_mwh == 0), None)
    out["lowest_battery_cost_without_battery"] = flips

    sweep = []
    for kwh in (100, 150, 200, 250, 300, 350, 400, 500):
        for km in (60_000, 120_000, 240_000):
            b = replace(a, battery_capex_eur_kwh=float(kwh), cable_capex_eur_km=float(km))
            sweep.append({"battery_eur_kwh": kwh, "cable_eur_km": km, **summary(with_capex(op, plans, b, f), plans, probs)})
    pd.DataFrame(sweep).round(1).to_csv(RES / "capex_sweep_full_year.csv", index=False)

    # Same plans on representative days, to show what the sensitivities miss.
    base = np.load(RES / "cost_table.npz")
    Q_rep = with_capex(base["op"], plans, a, f)
    out["representative_days_error"] = {
        "expected_cost_keur": {p.name: [float(dc.expected(Q_rep, probs)[i] / 1e3), float(e[i] / 1e3)]
                               for i, p in enumerate(plans) if p.name in ("nothing", "trunk", "8MWh@upstream", "8MWh@far_wind", "2MWh@far_wind+trunk")},
        "best_plan_representative": plans[int(dc.expected(Q_rep, probs).argmin())].name,
        "evpi_representative": dc.evpi(Q_rep, probs),
    }
    (RES / "summary_full_year.json").write_text(json.dumps(out, indent=2))

    # A setpoint of 1.00 is the cheap alternative, so it needs the AC check as well, at high load for low voltage.
    low = replace(a, v_slack_pu=1.00)
    by_name = {p.name: p for p in plans}
    ac = []
    for name in ("nothing", "trunk", "8MWh@far_wind"):
        plan = by_name[name]
        model = DayModel(f, plan, low)
        real = f.reinforced(plan.lines)
        net = build_ac_net(real, low)
        for s in [(1, 1, 2, 1), (2, 0, 2, 0), (0, 1, 1, 2)]:
            stats = [check_day(net, real, plan, low, *day_arrays(inp, s[0], s[1], s[2], day, s[3]), model) for day in range(0, 365, 3)]
            cat = {k: np.concatenate([x[k] for x in stats]) for k in stats[0]}
            ac.append({"plan": name, "scenario": str(s), "hours": len(cat["v_err"]), "v_err_max": cat["v_err"].max(),
                       "v_max": cat["v_max"].max(), "v_min": cat["v_min"].min(),
                       "hours_over_v": int((cat["v_max"] > low.v_max_pu + 1e-4).sum()),
                       "hours_under_v": int((cat["v_min"] < low.v_min_pu - 1e-4).sum()),
                       "loading_max_pct": cat["loading"].max(),
                       "hours_over_100pct": int((cat["loading"] > 100.0 + 1e-3).sum())})
    pd.DataFrame(ac).round(4).to_csv(RES / "ac_check_setpoint_100.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
