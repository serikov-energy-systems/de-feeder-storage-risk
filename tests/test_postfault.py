import numpy as np
import pytest

from feederrisk import postfault as pf
from feederrisk.config import Assumptions
from feederrisk.network import Feeder


def ring_feeder():
    """Substation - 0 - 1 with two branches: 1 - 2 - 3 and 1 - 4 - 5; leaf 6 hangs on bus 1. Ends 3 and 5 can be tied."""
    parent = np.array([-1, 0, 1, 2, 1, 4, 1])
    n = 7
    length = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 0.5])
    return Feeder(parent, 0.2 * length, 0.1 * length, np.full(n, 10.0), length, np.full(n, 0.1), np.zeros(n),
                  np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, -1))


def test_tie_buses_are_the_far_ends_of_separate_branches():
    f = ring_feeder()
    assert set(pf.tie_buses(f)) == {3, 5}


def test_fault_without_tie_isolates_exactly_the_subtree():
    f = ring_feeder()
    t = pf.post_fault_tree(f, fault=2)
    assert set(np.flatnonzero(~t.reach)) == {2, 3}
    t = pf.post_fault_tree(f, fault=0)
    assert not t.reach.any()


def test_tie_restores_branch_faults_but_not_the_head_or_a_side_leaf():
    f = ring_feeder()
    tie = pf.tie_buses(f)
    assert pf.post_fault_tree(f, fault=2, tie=tie).reach.all()
    assert pf.post_fault_tree(f, fault=4, tie=tie).reach.all()
    assert not pf.post_fault_tree(f, fault=6, tie=tie).reach[6]
    assert not pf.post_fault_tree(f, fault=0, tie=tie).reach.any()


def test_parallel_circuit_keeps_the_line_in_service_with_one_cable():
    f = ring_feeder()
    t = pf.post_fault_tree(f, doubled=(2,), fault=2)
    assert t.reach.all()
    assert t.r[2] == pytest.approx(f.r_ohm[2]) and t.s[2] == pytest.approx(f.s_max_mva[2])
    t = pf.post_fault_tree(f, doubled=(2,))
    assert t.r[2] == pytest.approx(f.r_ohm[2] / 2) and t.s[2] == pytest.approx(f.s_max_mva[2] * 2)


def test_linear_flow_voltage_matches_formula():
    f = ring_feeder()
    a = Assumptions()
    p = np.zeros((1, 7))
    p[0, 3] = 2.0
    v, s = pf.linear_flow(pf.post_fault_tree(f), p, np.zeros((1, 7)), a)
    drop = 2 * (f.r_ohm[0] + f.r_ohm[1] + f.r_ohm[2] + f.r_ohm[3]) * 2.0 / a.v_nominal_kv ** 2
    assert v[0, 3] == pytest.approx(np.sqrt(a.v_slack_pu ** 2 - drop))
    assert s[0, 0] == pytest.approx(2.0)


def test_reliability_cost_is_zero_without_faults_and_linear_in_voll():
    f = ring_feeder()
    a = Assumptions()
    load = np.full((48, 7), 0.1)
    gen = np.zeros((48, 7))
    none = pf.reliability(f, (), False, load, gen, a, pf.Reliability(fault_per_km_y=0.0))
    assert none["cost_eur"] == 0.0
    r1 = pf.reliability(f, (), False, load, gen, a, pf.Reliability(voll_eur_mwh=1000.0))
    r2 = pf.reliability(f, (), False, load, gen, a, pf.Reliability(voll_eur_mwh=2000.0))
    assert r2["cost_eur"] == pytest.approx(2 * r1["cost_eur"])
    assert r1["cost_eur"] == pytest.approx(pf.cost_from_parts(r1["ens_mwh"], r1["generation_lost_mwh"], a, 1000.0))


def test_tie_and_parallel_circuits_reduce_energy_not_supplied():
    f = ring_feeder()
    a, rel = Assumptions(), pf.Reliability()
    load = np.full((48, 7), 0.1)
    gen = np.zeros((48, 7))
    base = pf.reliability(f, (), False, load, gen, a, rel)["ens_mwh"]
    tied = pf.reliability(f, (), True, load, gen, a, rel)["ens_mwh"]
    doubled = pf.reliability(f, tuple(range(7)), False, load, gen, a, rel)["ens_mwh"]
    assert tied < base
    assert doubled == pytest.approx(0.0)


def test_overloaded_tie_does_not_restore():
    f = ring_feeder()
    a, rel = Assumptions(), pf.Reliability()
    load = np.full((4, 7), 8.0)               # far too much load for the tie and the other branch
    gen = np.zeros((4, 7))
    o = pf.fault_outcome(f, (), True, 2, load, gen, a, rel)
    assert not o["restored"].any()
    assert (o["interrupted_h"] == rel.repair_h).all()
