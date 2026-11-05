"""One day of operation on a feeder as a linear programme with linearized branch flow (LinDistFlow).

Network losses are neglected and reactive flows are taken as fixed, so thermal and voltage limits
reduce to variable bounds and sparse equalities. The AC check in accheck.py measures what this costs.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .config import Assumptions
from .network import Feeder, subtree_sums

HOURS = 24


@dataclass(frozen=True)
class Plan:
    name: str
    bus: int  # battery bus, -1 for none
    energy_mwh: float
    lines: tuple[int, ...] = ()  # reinforced lines

    def capex_annual(self, a: Assumptions, length_km: np.ndarray) -> float:
        km = float(length_km[list(self.lines)].sum()) if self.lines else 0.0
        return self.energy_mwh * a.battery_annual_eur_per_mwh() + km * a.cable_annual_eur_per_km()


@dataclass
class DayResult:
    cost: float
    arbitrage: float
    curtailed_mwh: float
    shed_mwh: float
    both_mwh: float  # energy charged and discharged in the same hour (should be small)
    p: np.ndarray
    v: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    curtail: np.ndarray
    shed: np.ndarray


class DayModel:
    """Sparse matrix for one feeder and plan, reused across days and scenarios."""

    def __init__(self, feeder: Feeder, plan: Plan, a: Assumptions):
        self.f = feeder.reinforced(plan.lines)
        self.plan = plan
        self.a = a
        self.n = self.f.n
        self.nv = 4 * self.n + 3  # p, v, curtail, shed, charge, discharge, soc per hour
        self.total = self.nv * HOURS
        self.vbase = a.v_nominal_kv**2
        self.kids = self.f.children()
        self.A, self.rows_soc = self._matrix()

    def _i(self, t: int, kind: str, bus: int = 0) -> int:
        base = t * self.nv
        n = self.n
        offset = {"p": bus, "v": n + bus, "c": 2 * n + bus, "s": 3 * n + bus,
                  "ch": 4 * n, "dis": 4 * n + 1, "soc": 4 * n + 2}[kind]
        return base + offset

    def _matrix(self):
        rows, cols, vals = [], [], []
        row = 0

        def put(r, c, v):
            rows.append(r)
            cols.append(c)
            vals.append(v)

        for t in range(HOURS):
            for i in range(self.n):
                # flow into bus - flow onward + shed - curtail - charge + discharge = load - generation
                put(row, self._i(t, "p", i), 1.0)
                for k in self.kids[i]:
                    put(row, self._i(t, "p", k), -1.0)
                put(row, self._i(t, "s", i), 1.0)
                put(row, self._i(t, "c", i), -1.0)
                if i == self.plan.bus:
                    put(row, self._i(t, "ch"), -1.0)
                    put(row, self._i(t, "dis"), 1.0)
                row += 1
            for i in range(self.n):
                # v_i = v_parent - 2 (r p + x q) / V^2, squared voltage in per unit
                put(row, self._i(t, "v", i), 1.0)
                par = self.f.parent[i]
                if par >= 0:
                    put(row, self._i(t, "v", par), -1.0)
                put(row, self._i(t, "p", i), 2 * self.f.r_ohm[i] / self.vbase)
                row += 1
        rows_soc = row
        for t in range(HOURS):
            put(row, self._i(t, "soc"), 1.0)
            if t > 0:
                put(row, self._i(t - 1, "soc"), -1.0)
            put(row, self._i(t, "ch"), -self.a.eta)
            put(row, self._i(t, "dis"), 1.0 / self.a.eta)
            row += 1
        A = sp.csr_matrix((vals, (rows, cols)), shape=(row, self.total))
        return A, rows_soc

    def solve(self, load: np.ndarray, gen: np.ndarray, price: np.ndarray) -> DayResult:
        """load and gen have shape (24, n) in MW; price has shape (24,) in EUR/MWh."""
        a, n, f = self.a, self.n, self.f
        q_bus = load * f.load_q_share
        q_flow = subtree_sums(f, q_bus)
        p_lim = np.sqrt(np.maximum(f.s_max_mva**2 - q_flow**2, 0.0))
        v_min, v_max = a.v_min_pu**2, a.v_max_pu**2
        v0 = a.v_slack_pu**2
        energy = self.plan.energy_mwh
        power = energy / a.duration_h

        lb = np.zeros(self.total)
        ub = np.zeros(self.total)
        c = np.zeros(self.total)
        b = np.zeros(self.A.shape[0])
        for t in range(HOURS):
            s = t * self.nv
            lb[s : s + n] = -p_lim[t]
            ub[s : s + n] = p_lim[t]
            lb[s + n : s + 2 * n] = v_min
            ub[s + n : s + 2 * n] = v_max
            ub[s + 2 * n : s + 3 * n] = gen[t]
            ub[s + 3 * n : s + 4 * n] = load[t]
            ub[s + 4 * n] = power
            ub[s + 4 * n + 1] = power
            ub[s + 4 * n + 2] = energy
            c[s + 2 * n : s + 3 * n] = a.curtail_eur_mwh
            c[s + 3 * n : s + 4 * n] = a.voll_eur_mwh
            c[s + 4 * n] = price[t]
            c[s + 4 * n + 1] = -price[t] + a.wear_eur_mwh
            r = 2 * n * t
            b[r : r + n] = load[t] - gen[t]
            drop = 2 * f.x_ohm * q_flow[t] / self.vbase
            b[r + n : r + 2 * n] = -drop + np.where(f.parent < 0, v0, 0.0)
        # The day starts and ends half full, so days can be solved independently.
        b[self.rows_soc] = 0.5 * energy
        last = (HOURS - 1) * self.nv + 4 * n + 2
        lb[last] = ub[last] = 0.5 * energy

        res = linprog(c, A_eq=self.A, b_eq=b, bounds=np.column_stack([lb, ub]), method="highs")
        if res.status != 0:
            raise RuntimeError(f"day model not solved: {res.message}")
        x = res.x.reshape(HOURS, self.nv)
        ch, dis = x[:, 4 * n], x[:, 4 * n + 1]
        return DayResult(
            cost=float(res.fun),
            arbitrage=float(np.sum(price * (dis - ch))),
            curtailed_mwh=float(x[:, 2 * n : 3 * n].sum()),
            shed_mwh=float(x[:, 3 * n : 4 * n].sum()),
            both_mwh=float(np.minimum(ch, dis).sum()),
            p=x[:, :n],
            v=x[:, n : 2 * n],
            charge=ch,
            discharge=dis,
            curtail=x[:, 2 * n : 3 * n],
            shed=x[:, 3 * n : 4 * n],
        )
