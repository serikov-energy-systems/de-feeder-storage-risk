"""AC power flow check of the linearized plans with pandapower."""

import numpy as np

from .config import Assumptions
from .daylp import DayModel, Plan
from .network import Feeder


def build_ac_net(f: Feeder, a: Assumptions):
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=1.0)
    root = pp.create_bus(net, a.v_nominal_kv)
    pp.create_ext_grid(net, root, vm_pu=a.v_slack_pu)
    buses = [pp.create_bus(net, a.v_nominal_kv) for _ in range(f.n)]
    for i in range(f.n):
        frm = root if f.parent[i] < 0 else buses[f.parent[i]]
        ka = f.s_max_mva[i] / (a.v_nominal_kv * np.sqrt(3))
        pp.create_line_from_parameters(net, frm, buses[i], length_km=1.0, r_ohm_per_km=f.r_ohm[i],
                                       x_ohm_per_km=f.x_ohm[i], c_nf_per_km=0.0, max_i_ka=ka)
    for b in buses:
        pp.create_load(net, b, p_mw=0.0, q_mvar=0.0)
    return net


def check_day(net, f: Feeder, plan: Plan, a: Assumptions, load, gen, price, model: DayModel) -> dict:
    """Re-solve one day with the LP, then run AC power flow on each hour of its dispatch."""
    import pandapower as pp

    r = model.solve(load, gen, price)
    out = []
    for t in range(load.shape[0]):
        p = load[t] - r.shed[t] - gen[t] + r.curtail[t]
        if plan.bus >= 0:
            p = p.copy()
            p[plan.bus] += r.charge[t] - r.discharge[t]
        net.load["p_mw"] = p
        net.load["q_mvar"] = load[t] * f.load_q_share
        pp.runpp(net, numba=False)
        v_ac = net.res_bus.vm_pu.to_numpy()[1:]
        v_lin = np.sqrt(r.v[t])
        out.append({
            "v_err": float(np.abs(v_ac - v_lin).max()),
            "v_max": float(v_ac.max()),
            "v_min": float(v_ac.min()),
            "loading": float(net.res_line.loading_percent.max()),
            "loss_mw": float(net.res_line.pl_mw.sum()),
            "feed_mw": float(abs(net.res_ext_grid.p_mw.iloc[0])),
        })
    return {k: np.array([o[k] for o in out]) for k in out[0]}
