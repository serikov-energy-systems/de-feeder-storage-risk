import numpy as np
import pytest

from feederrisk.config import Assumptions
from feederrisk.daylp import DayModel, Plan
from feederrisk.network import Feeder, subtree_sums
from feederrisk.representative import select_days


def small_feeder(s_max=10.0):
    # Substation - 0 - 1 - 2, with a side branch 3 at bus 0.
    parent = np.array([-1, 0, 1, 0])
    n = 4
    return Feeder(parent, np.full(n, 1.0), np.full(n, 0.5), np.full(n, s_max), np.ones(n), np.ones(n),
                  np.zeros(n), np.array([0.0, 0.0, 5.0, 0.0]), np.zeros(n), np.zeros(n), np.full(n, -1))


def test_subtree_sums():
    f = small_feeder()
    assert subtree_sums(f, np.array([1.0, 2.0, 3.0, 4.0])).tolist() == [10.0, 5.0, 3.0, 4.0]


def test_flows_balance_and_voltage_matches_formula():
    f = small_feeder()
    a = Assumptions(v_min_pu=0.5, v_max_pu=1.5)
    m = DayModel(f, Plan("none", -1, 0.0), a)
    load = np.tile([1.0, 1.0, 1.0, 1.0], (24, 1))
    r = m.solve(load, np.zeros((24, 4)), np.full(24, 50.0))
    assert r.cost == pytest.approx(0.0)
    assert r.p[0].tolist() == pytest.approx([4.0, 2.0, 1.0, 1.0])
    v1 = a.v_slack_pu**2 - 2 * 1.0 * 4.0 / 400
    assert r.v[0][0] == pytest.approx(v1)


def test_thermal_limit_forces_curtailment():
    f = small_feeder(s_max=3.0)
    m = DayModel(f, Plan("none", -1, 0.0), Assumptions())
    gen = np.zeros((24, 4))
    gen[:, 2] = 5.0
    r = m.solve(np.zeros((24, 4)), gen, np.full(24, 50.0))
    assert r.curtailed_mwh == pytest.approx(2.0 * 24)
    assert r.cost == pytest.approx(2.0 * 24 * Assumptions().curtail_eur_mwh)


def test_battery_downstream_reduces_curtailment_but_upstream_does_not():
    f = small_feeder(s_max=3.0)
    a = Assumptions()
    gen = np.zeros((24, 4))
    gen[10:14, 2] = 5.0  # four hours over the limit of line 2
    price = np.full(24, 50.0)
    load = np.zeros((24, 4))
    at_gen = DayModel(f, Plan("gen", 2, 4.0), a).solve(load, gen, price)
    behind = DayModel(f, Plan("side", 3, 4.0), a).solve(load, gen, price)
    assert at_gen.curtailed_mwh < behind.curtailed_mwh - 1.0
    assert behind.curtailed_mwh == pytest.approx(8.0)


def test_battery_energy_balance_and_arbitrage():
    f = small_feeder()
    a = Assumptions(v_min_pu=0.5, v_max_pu=1.5)
    m = DayModel(f, Plan("b", 0, 4.0), a)
    price = np.r_[np.full(12, 10.0), np.full(12, 100.0)]
    r = m.solve(np.zeros((24, 4)), np.zeros((24, 4)), price)
    assert r.arbitrage > 0
    assert r.charge.sum() * a.eta == pytest.approx(r.discharge.sum() / a.eta)
    assert r.charge.max() <= 2.0 + 1e-9 and r.discharge.max() <= 2.0 + 1e-9
    assert r.both_mwh == pytest.approx(0.0, abs=1e-9)


def test_reinforcement_halves_impedance_doubles_rating():
    f = small_feeder().reinforced((0,))
    assert f.r_ohm[0] == 0.5 and f.x_ohm[0] == 0.25 and f.s_max_mva[0] == 20.0 and f.r_ohm[1] == 1.0


def test_day_weights_sum_to_year_and_forced_day_counts_once():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(365, 8))
    x[100] += 50
    days, weights = select_days(x, 10, [100])
    assert weights.sum() == 365
    assert 100 in days.tolist()
    assert weights[days.tolist().index(100)] == 1
