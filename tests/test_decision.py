import numpy as np
import pytest

from feederrisk import decision as dc


def table():
    # Two plans, one source with two states: plan 0 suits state 0, plan 1 suits state 1.
    return np.array([[0.0, 10.0], [6.0, 2.0]]), [np.array([0.5, 0.5])]


def test_evpi_hand_calculation():
    Q, p = table()
    # Plan costs 5 and 4 in expectation; knowing the state gives 0 and 2, mean 1.
    assert dc.expected(Q, p).tolist() == [5.0, 4.0]
    assert dc.evpi(Q, p) == pytest.approx(3.0)


def test_evsi_limits():
    Q, p = table()
    assert dc.evsi(Q, p, 0, 1.0) == pytest.approx(dc.evpi(Q, p))
    assert dc.evsi(Q, p, 0, 0.5) == pytest.approx(0.0)


def test_cvar_splits_boundary_atom():
    costs = np.array([0.0, 10.0, 20.0, 30.0])
    w = np.full(4, 0.25)
    assert dc.cvar(costs, w, 0.5) == pytest.approx(25.0)
    assert dc.cvar(costs, w, 0.9) == pytest.approx(30.0)
    # Worst 40%: all of 30 and 0.15 of 20.
    assert dc.cvar(costs, w, 0.6) == pytest.approx((0.25 * 30 + 0.15 * 20) / 0.4)


def test_partial_evpi_never_exceeds_total_and_is_nonnegative():
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(6, 3, 3, 2))
    p = [np.array([0.2, 0.3, 0.5]), np.array([0.25, 0.5, 0.25]), np.array([0.5, 0.5])]
    total = dc.evpi(Q, p)
    for ax in range(3):
        v = dc.evpi(Q, p, (ax,))
        assert -1e-12 <= v <= total + 1e-12
    assert dc.evpi(Q, p, (0, 1, 2)) == pytest.approx(total)


def test_vss_zero_when_central_plan_is_best():
    Q, p = table()
    assert dc.vss(Q, p, (1,)) == pytest.approx(0.0)
    assert dc.vss(Q, p, (0,)) == pytest.approx(1.0)
