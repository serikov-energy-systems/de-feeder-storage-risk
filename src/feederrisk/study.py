"""Plan set, scenario set and the cost table Q(plan, weather, price, DER, load)."""

from dataclasses import dataclass, replace
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import BATTERY_SIZES_MWH, DER_LEVELS, LEVEL_PROB, LOAD_LEVELS, Assumptions
from .daylp import DayModel, Plan
from .network import Feeder
from .representative import select_days

SOURCES = ("weather", "price", "der", "load")
HOURS = 24
WINDIEST_DAYS = 60


@dataclass
class Inputs:
    feeder: Feeder
    load_h: np.ndarray
    wind_cf: np.ndarray  # (weather, hour)
    solar_cf: np.ndarray
    price: np.ndarray  # (price year, hour)
    weather_years: np.ndarray
    extras: dict


def load_inputs(path: Path) -> Inputs:
    d = np.load(path)
    feeder = Feeder(d["parent"], d["r_ohm"], d["x_ohm"], d["s_max_mva"], d["length_km"], d["load_p_mw"],
                    d["load_q_share"], d["wind_mw"], d["pv_mw"], d["flat_mw"], d["bus_id"])
    return Inputs(feeder, d["load_h"], d["wind_cf"], d["solar_cf"], d["price"], d["weather_years"],
                  {"all_years_cf": d["all_years_cf"], "weather_annual_cf": d["weather_annual_cf"]})


def scenario_probs(n_weather: int, n_price: int) -> list[np.ndarray]:
    level = np.array([LEVEL_PROB[k] for k in ("low", "mid", "high")])
    return [np.full(n_weather, 1 / n_weather), np.full(n_price, 1 / n_price), level, level]


CENTRAL = (1, 1, 1, 1)


def build_plans(f: Feeder) -> list[Plan]:
    """No action, battery at candidate buses and sizes, parallel cable to the wind buses, and combinations."""
    wind_buses = [int(i) for i in np.nonzero(f.wind_mw)[0]]
    far_wind = max(wind_buses, key=lambda b: f.r_ohm[f.path_to_root(b)].sum())
    near_wind = min(wind_buses, key=lambda b: f.r_ohm[f.path_to_root(b)].sum())
    wind_path = set(f.path_to_root(far_wind)) | set(f.path_to_root(near_wind))
    leaf = max((b for b in range(f.n) if b not in wind_path), key=lambda b: f.r_ohm[f.path_to_root(b)].sum())
    shared = [b for b in f.path_to_root(far_wind) if b in f.path_to_root(near_wind)]
    upstream = int(f.parent[near_wind])
    # Wind buses, the bus just upstream of them, the head of the feeder and the far end of the branch without wind.
    candidates = {"far_wind": far_wind, "near_wind": near_wind, "upstream": upstream, "head": 0, "leaf": leaf}
    if len(set(candidates.values())) < len(candidates):
        raise ValueError(f"candidate buses coincide: {candidates}")
    trunk = tuple(sorted(shared))
    all_wind = tuple(sorted(set(f.path_to_root(far_wind)) | set(f.path_to_root(near_wind))))
    plans = [Plan("nothing", -1, 0.0)]
    for lines, label in (((), ""), (trunk, "+trunk"), (all_wind, "+wind_lines")):
        if lines:
            plans.append(Plan(label[1:], -1, 0.0, lines))
        for (name, bus), size in product(candidates.items(), BATTERY_SIZES_MWH):
            if lines and name in ("head", "leaf"):
                continue
            plans.append(Plan(f"{size:g}MWh@{name}{label}", bus, size, lines))
    return plans


def representative_days(inp: Inputs, w: int, p: int, k: int = 18):
    days = lambda a: a.reshape(365, HOURS)
    wind, solar, price = days(inp.wind_cf[w]), days(inp.solar_cf[w]), days(inp.price[p])
    load = days(inp.load_h.sum(axis=1))
    feats = np.hstack([wind, solar, price / 50.0, load / load.max()])
    spread = price.max(axis=1) - price.min(axis=1)
    # Curtailment sits in the windiest days, so those are solved one by one with weight one.
    windiest = np.argsort(wind.max(axis=1))[-WINDIEST_DAYS:]
    forced = sorted({*map(int, windiest), int(wind.sum(axis=1).argmax()),
                     int(price.min(axis=1).argmin()), int(spread.argmax()),
                     int(solar.sum(axis=1).argmax()), int(load.max(axis=1).argmax())})
    return select_days(feats, k, forced)


def day_arrays(inp: Inputs, w: int, p: int, d: int, day: int, l: int):
    """Hourly load, generation and price for one calendar day of one scenario."""
    f = inp.feeder
    der = list(DER_LEVELS.values())[d]
    ls = list(LOAD_LEVELS.values())[l]
    idx = slice(day * HOURS, (day + 1) * HOURS)
    load = inp.load_h[idx] * ls
    gen = der * (np.outer(inp.wind_cf[w][idx], f.wind_mw) + np.outer(inp.solar_cf[w][idx], f.pv_mw)) + f.flat_mw
    return load, gen, inp.price[p][idx]


def blind(a: Assumptions, f: Feeder) -> tuple[Assumptions, Feeder]:
    """Copper plate: the same model with network limits removed."""
    return replace(a, v_min_pu=0.0, v_max_pu=10.0), replace(f, s_max_mva=f.s_max_mva * 1000)


def _plan_table(args):
    inp, plan, a, rep, shape = args
    model = DayModel(inp.feeder, plan, a)
    op = np.zeros(shape)
    parts = np.zeros((*shape, 4))
    for w, p, d, l in product(*(range(s) for s in shape)):
        days, weights = rep[(w, p)]
        for day, wt in zip(days, weights):
            load, gen, price = day_arrays(inp, w, p, d, int(day), l)
            r = model.solve(load, gen, price)
            op[w, p, d, l] += wt * r.cost
            parts[w, p, d, l] += wt * np.array([r.arbitrage, r.curtailed_mwh, r.shed_mwh, r.both_mwh])
    return op, parts


def cost_table(inp: Inputs, plans: list[Plan], a: Assumptions, workers: int = 8):
    """Operating cost by plan and scenario, and its parts (arbitrage, curtailed MWh, shed MWh, overlap MWh)."""
    shape = (inp.wind_cf.shape[0], inp.price.shape[0], len(DER_LEVELS), len(LOAD_LEVELS))
    rep = {(w, p): representative_days(inp, w, p) for w in range(shape[0]) for p in range(shape[1])}
    jobs = [(inp, plan, a, rep, shape) for plan in plans]
    with Pool(workers) as pool:
        out = pool.map(_plan_table, jobs, chunksize=1)
    op = np.stack([o[0] for o in out])
    parts = np.stack([o[1] for o in out])
    return op, parts, rep


def _year_cost(args):
    inp, plan, a, scen = args
    w, p, d, l = scen
    model = DayModel(inp.feeder, plan, a)
    total = 0.0
    for day in range(365):
        total += model.solve(*day_arrays(inp, w, p, d, day, l)).cost
    return total


def full_year_costs(inp, plans, a, scenarios, workers: int = 8):
    jobs = [(inp, plan, a, s) for plan in plans for s in scenarios]
    with Pool(workers) as pool:
        vals = pool.map(_year_cost, jobs, chunksize=1)
    return np.array(vals).reshape(len(plans), len(scenarios))
