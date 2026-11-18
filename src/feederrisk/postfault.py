"""N-1 analysis: post-fault topologies, restoration by a parallel circuit or an open tie, and reliability cost.

After the outage of one line the feeder is again a tree (a parallel circuit stays in service with the impedance of
one cable; a normally open tie to the far end of the other branch is closed). Hourly voltage and thermal limits of
that tree are checked with linearized branch flow over all 8760 hours at once, allowing the renewable generation
to be curtailed in steps. Hours in which no step is feasible are hours in which supply is not restored before the
repair is finished. pandapower AC power flow checks the verdicts (see accheck.py and scripts/run_n1.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Assumptions
from .network import Feeder

ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)  # share of the renewable generation kept after a fault


@dataclass(frozen=True)
class Reliability:
    """Illustrative reliability inputs; they are assumptions, not data of any network."""

    fault_per_km_y: float = 0.05
    repair_h: float = 8.0
    switching_h: float = 1.0  # manual switching to the tie
    tie_km: float = 3.0
    voll_eur_mwh: float = 10_000.0


def tie_buses(f: Feeder) -> tuple[int, int]:
    """Far ends of the two branches that part earliest from each other: the open point of a ring."""
    kids = f.children()
    leaves = [b for b in range(f.n) if not kids[b]]
    dist = {b: float(f.length_km[f.path_to_root(b)].sum()) for b in leaves}
    a = max(leaves, key=dist.get)
    pa = f.path_to_root(a)[::-1]

    def lca_depth(b: int) -> int:
        pb = f.path_to_root(b)[::-1]
        k = 0
        while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
            k += 1
        return k

    others = [b for b in leaves if b != a]
    b = min(others, key=lambda x: (lca_depth(x), -dist[x]))
    return a, b


def subtree(f: Feeder, bus: int) -> list[int]:
    kids = f.children()
    out, stack = [], [bus]
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(kids[i])
    return out


@dataclass
class Tree:
    parent: np.ndarray  # -1 for buses fed from the substation, -2 for unreachable buses
    r: np.ndarray
    x: np.ndarray
    s: np.ndarray
    order: list[int]  # reachable buses, parents first

    @property
    def reach(self) -> np.ndarray:
        return self.parent > -2


def post_fault_tree(f: Feeder, doubled: tuple[int, ...] = (), fault: int | None = None,
                    tie: tuple[int, int] | None = None, tie_km: float = 3.0, tie_cable: int = 3) -> Tree:
    """Tree that supplies the feeder after the outage of `fault` (None: intact), closing `tie` if given.

    `f` is the original feeder; lines in `doubled` have a parallel circuit, so a fault leaves one circuit.
    """
    n = f.n
    r, x, s = f.r_ohm.copy(), f.x_ohm.copy(), f.s_max_mva.copy()
    r[list(doubled)] /= 2
    x[list(doubled)] /= 2
    s[list(doubled)] *= 2
    edges = []  # (u, v, r, x, s); the substation is node n
    for i in range(n):
        p = int(f.parent[i])
        u = n if p < 0 else p
        if fault is not None and i == fault:
            if i in doubled:
                edges.append((u, i, f.r_ohm[i], f.x_ohm[i], f.s_max_mva[i]))  # one circuit of the pair
            continue
        edges.append((u, i, r[i], x[i], s[i]))
    if tie is not None:
        per_km = lambda v: v[tie_cable] / f.length_km[tie_cable]
        edges.append((tie[0], tie[1], per_km(f.r_ohm) * tie_km, per_km(f.x_ohm) * tie_km, f.s_max_mva[tie_cable]))
    adj: dict[int, list] = {k: [] for k in range(n + 1)}
    for u, v, rr, xx, ss in edges:
        adj[u].append((v, rr, xx, ss))
        adj[v].append((u, rr, xx, ss))
    parent = np.full(n, -2)
    pr, px, ps = np.zeros(n), np.zeros(n), np.zeros(n)
    seen, order, queue = {n}, [], [n]
    while queue:
        u = queue.pop(0)
        for v, rr, xx, ss in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            parent[v] = -1 if u == n else u
            pr[v], px[v], ps[v] = rr, xx, ss
            order.append(v)
            queue.append(v)
    return Tree(parent, pr, px, ps, order)


def linear_flow(tree: Tree, p: np.ndarray, q: np.ndarray, a: Assumptions) -> tuple[np.ndarray, np.ndarray]:
    """Bus voltage (pu) and branch apparent power (MVA) of a tree for net injections p, q of shape (T, n)."""
    n = len(tree.parent)
    P = np.where(tree.reach, p, 0.0).astype(float)
    Q = np.where(tree.reach, q, 0.0).astype(float)
    for i in reversed(tree.order):
        par = tree.parent[i]
        if par >= 0:
            P[:, par] += P[:, i]
            Q[:, par] += Q[:, i]
    v2 = np.full(p.shape, a.v_slack_pu ** 2)
    vbase = a.v_nominal_kv ** 2
    for i in tree.order:
        par = tree.parent[i]
        up = a.v_slack_pu ** 2 if par < 0 else v2[:, par]
        v2[:, i] = up - 2 * (tree.r[i] * P[:, i] + tree.x[i] * Q[:, i]) / vbase
    return np.sqrt(np.maximum(v2, 0.0)), np.hypot(P, Q)


def feasible(tree: Tree, p: np.ndarray, q: np.ndarray, a: Assumptions) -> np.ndarray:
    """Hours in which all reachable buses and lines stay within limits."""
    v, s = linear_flow(tree, p, q, a)
    r = tree.reach
    ok_v = ((v[:, r] >= a.v_min_pu) & (v[:, r] <= a.v_max_pu)).all(axis=1)
    ok_s = (s[:, r] <= tree.s[r] + 1e-9).all(axis=1)
    return ok_v & ok_s


def scenario_arrays(f: Feeder, load_h, wind_cf, solar_cf, der: float, load_scale: float):
    """Hourly load and renewable generation of a scenario, shape (8760, n)."""
    load = load_h * load_scale
    gen = der * (np.outer(wind_cf, f.wind_mw) + np.outer(solar_cf, f.pv_mw)) + f.flat_mw
    return load, gen


def fault_outcome(f: Feeder, doubled: tuple[int, ...], tie_on: bool, line: int, load, gen, a: Assumptions,
                  rel: Reliability) -> dict:
    """Hour-by-hour outcome of a fault on `line`: restoration, curtailment and the cost of one fault."""
    n = f.n
    q = load * f.load_q_share
    tie = tie_buses(f) if tie_on else None
    S = subtree(f, line)
    is_doubled = line in doubled
    tree = post_fault_tree(f, doubled, line, tie, rel.tie_km)
    covered = bool(tree.reach[S].all())               # all buses of the subtree are fed again
    best = np.full(load.shape[0], -1.0)               # largest feasible generation share, -1 if none
    if covered:
        for alpha in ALPHAS:
            ok = feasible(tree, load - alpha * gen, q, a)
            best = np.where((best < 0) & ok, alpha, best)
    restored = best >= 0
    load_s, gen_s = load[:, S].sum(axis=1), gen[:, S].sum(axis=1)
    gen_all = gen.sum(axis=1)
    tau = 0.0 if is_doubled else rel.switching_h
    R = rel.repair_h
    curtail = a.curtail_eur_mwh
    interrupted_h = np.where(restored, tau, R)                                   # hours without supply
    ens = load_s * interrupted_h                                                 # MWh
    lost_gen = gen_s * interrupted_h
    curtailed = np.where(restored, (1 - np.where(restored, best, 1.0)) * gen_all * (R - tau), 0.0)
    cost = rel.voll_eur_mwh * ens + curtail * (lost_gen + curtailed)
    return {"cost": cost, "ens": ens, "restored": restored, "curtailed_mwh": curtailed, "interrupted_h": interrupted_h,
            "covered": covered, "lost_gen_mwh": lost_gen, "best_alpha": best}


def reliability(f: Feeder, doubled: tuple[int, ...], tie_on: bool, load, gen, a: Assumptions,
                rel: Reliability) -> dict:
    """Expected annual reliability cost (EUR), energy not supplied (MWh) and curtailed energy (MWh)."""
    cost = ens = curt = lost = 0.0
    per_line = []
    for line in range(f.n):
        o = fault_outcome(f, doubled, tie_on, line, load, gen, a, rel)
        rate = rel.fault_per_km_y * f.length_km[line]
        cost += rate * o["cost"].mean()
        ens += rate * o["ens"].mean()
        curt += rate * o["curtailed_mwh"].mean()
        lost += rate * (o["curtailed_mwh"] + o["lost_gen_mwh"]).mean()
        per_line.append({"line": line, "fault_per_year": rate, "restored_hours_share": float(o["restored"].mean()),
                         "ens_mwh_per_fault": float(o["ens"].mean())})
    return {"cost_eur": cost, "ens_mwh": ens, "curtailed_mwh": curt, "generation_lost_mwh": lost,
            "per_line": per_line}


def cost_from_parts(ens_mwh, generation_lost_mwh, a: Assumptions, voll_eur_mwh: float, fault_scale: float = 1.0):
    """Reliability cost for other VOLL or fault rates: it is linear in both."""
    return fault_scale * (voll_eur_mwh * ens_mwh + a.curtail_eur_mwh * generation_lost_mwh)
