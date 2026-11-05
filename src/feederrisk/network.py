"""Radial feeder as plain arrays, so that the optimization needs no power-system library."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Feeder:
    """Buses 0..n-1 exclude the substation; line i feeds bus i from parent[i] (-1 is the substation)."""

    parent: np.ndarray
    r_ohm: np.ndarray
    x_ohm: np.ndarray
    s_max_mva: np.ndarray
    length_km: np.ndarray
    load_p_mw: np.ndarray  # today's peak of each bus, used for reporting
    load_q_share: np.ndarray  # Q/P per bus
    wind_mw: np.ndarray
    pv_mw: np.ndarray
    flat_mw: np.ndarray
    bus_id: np.ndarray  # SimBench bus index, -1 in synthetic feeders

    @property
    def n(self) -> int:
        return len(self.parent)

    def children(self) -> list[list[int]]:
        kids: list[list[int]] = [[] for _ in range(self.n)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(i)
        return kids

    def path_to_root(self, bus: int) -> list[int]:
        path = []
        while bus >= 0:
            path.append(bus)
            bus = int(self.parent[bus])
        return path

    def reinforced(self, lines: tuple[int, ...]) -> "Feeder":
        """Parallel cable of the same type: half the impedance, twice the rating."""
        if not lines:
            return self
        idx = list(lines)
        r, x, s = self.r_ohm.copy(), self.x_ohm.copy(), self.s_max_mva.copy()
        r[idx] /= 2
        x[idx] /= 2
        s[idx] *= 2
        return replace(self, r_ohm=r, x_ohm=x, s_max_mva=s)


def subtree_sums(feeder: Feeder, bus_values: np.ndarray) -> np.ndarray:
    """Sum of bus values at and below each bus. Works along the last axis."""
    out = bus_values.astype(float).copy()
    order = np.argsort([len(feeder.path_to_root(i)) for i in range(feeder.n)])[::-1]
    for i in order:
        p = feeder.parent[i]
        if p >= 0:
            out[..., p] += out[..., i]
    return out


def load_simbench_feeder(code: str, n_buses: int):
    """Feeder with n_buses of a SimBench net, with hourly per-bus load profiles.

    Returns the feeder, hourly load P of shape (8760, n) and the source pandapower net.
    """
    import networkx as nx
    import pandas as pd
    import simbench as sb

    net = sb.get_simbench_net(code)
    open_lines = set(net.switch[(net.switch.et == "l") & (~net.switch.closed)].element)
    lines = net.line[~net.line.index.isin(open_lines)]
    g = nx.Graph()
    for idx, row in lines.iterrows():
        g.add_edge(int(row.from_bus), int(row.to_bus), line=idx)
    root = int(net.trafo.lv_bus.iloc[0])
    h = g.copy()
    h.remove_node(root)
    comp = next(c for c in nx.connected_components(h) if len(c) == n_buses)
    tree = g.subgraph(comp | {root}).copy()
    if not nx.is_tree(tree):
        raise ValueError("feeder is not radial with the open switches respected")
    order = list(nx.bfs_tree(tree, root))[1:]
    pos = {b: i for i, b in enumerate(order)}
    pred = dict(nx.bfs_predecessors(tree, root))
    n = len(order)
    parent = np.array([pos.get(pred[b], -1) for b in order])
    r, x, s, length = (np.zeros(n) for _ in range(4))
    v_kv = 20.0
    for i, b in enumerate(order):
        ln = net.line.loc[tree[pred[b]][b]["line"]]
        par = ln.parallel
        r[i] = ln.r_ohm_per_km * ln.length_km / par
        x[i] = ln.x_ohm_per_km * ln.length_km / par
        s[i] = ln.max_i_ka * par * v_kv * np.sqrt(3)
        length[i] = ln.length_km

    prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    lp = prof[("load", "p_mw")]
    hours = np.arange(len(lp)) // 4
    load_h = np.zeros((hours.max() + 1, n))
    q_share = np.zeros(n)
    for li, row in net.load[net.load.bus.isin(comp)].iterrows():
        i = pos[int(row.bus)]
        load_h[:, i] += pd.Series(lp[li].to_numpy()).groupby(hours).mean().to_numpy()
        q_share[i] = row.q_mvar / row.p_mw if row.p_mw > 0 else 0.0
    # SimBench covers a leap year: remove 29 February to match 365-day weather and price years.
    keep = np.r_[0 : 59 * 24, 60 * 24 : load_h.shape[0]]
    load_h = load_h[keep][:8760]

    wind, pv, flat = (np.zeros(n) for _ in range(3))
    for _, row in net.sgen[net.sgen.bus.isin(comp)].iterrows():
        i = pos[int(row.bus)]
        name = str(row.type)
        if name.startswith("Wind"):
            wind[i] += row.p_mw
        elif name.startswith("PV") or name.startswith("lv_RES"):
            pv[i] += row.p_mw
        else:
            flat[i] += row.p_mw * 0.5  # biomass and hydro run steadily at half rating
    feeder = Feeder(parent, r, x, s, length, load_h.max(axis=0), q_share, wind, pv, flat, np.array(order))
    return feeder, load_h, net
