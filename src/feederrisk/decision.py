"""First-stage choice over a table of plan costs by scenario, and the value of information.

Q has shape (plans, *dims) with one axis per uncertainty source; probs are per axis and independent.
Because the plan set is finite, all of these quantities are computed exactly by enumeration.
"""

from __future__ import annotations

import numpy as np


def joint_probability(probs: list[np.ndarray]) -> np.ndarray:
    out = np.array(1.0)
    for p in probs:
        out = np.multiply.outer(out, p)
    return out


def expected(Q: np.ndarray, probs: list[np.ndarray]) -> np.ndarray:
    w = joint_probability(probs)
    return (Q * w).reshape(Q.shape[0], -1).sum(axis=1)


def cvar(costs: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Mean cost of the worst (1 - alpha) probability mass, splitting the boundary atom."""
    order = np.argsort(costs)[::-1]
    c, w = costs[order], weights[order]
    tail = 1.0 - alpha
    if tail <= 0:
        return float(c[0])
    cum = np.cumsum(w)
    k = int(np.searchsorted(cum, tail - 1e-12))
    taken = w[: k + 1].copy()
    taken[k] = tail - (cum[k - 1] if k > 0 else 0.0)
    return float(np.dot(c[: k + 1], taken) / tail)


def cvars(Q: np.ndarray, probs: list[np.ndarray], alpha: float) -> np.ndarray:
    w = joint_probability(probs).ravel()
    return np.array([cvar(Q[i].ravel(), w, alpha) for i in range(Q.shape[0])])


def risk_objective(Q, probs, beta: float, alpha: float = 0.9) -> np.ndarray:
    return (1 - beta) * expected(Q, probs) + beta * cvars(Q, probs, alpha)


def _marginal(Q, probs, keep: tuple[int, ...]) -> np.ndarray:
    """Expected cost of each plan given the sources in keep, other sources averaged out."""
    cond = Q
    for ax in sorted((i for i in range(len(probs)) if i not in keep), reverse=True):
        shape = [1] * cond.ndim
        shape[ax + 1] = -1
        cond = (cond * probs[ax].reshape(shape)).sum(axis=ax + 1)
    return cond


def evpi(Q, probs, axes: tuple[int, ...] | None = None) -> float:
    """Expected cost saved by learning the sources in axes before deciding (all sources if None)."""
    axes = tuple(range(len(probs))) if axes is None else tuple(sorted(axes))
    base = float(expected(Q, probs).min())
    best = _marginal(Q, probs, axes).min(axis=0)
    w = joint_probability([probs[i] for i in axes])
    return base - float((best * w).sum())


def evsi(Q, probs, axis: int, accuracy: float) -> float:
    """Value of a signal that names the true state of one source with this accuracy.

    A wrong signal points to one of the other states with equal chance.
    """
    p = probs[axis]
    k = len(p)
    like = np.full((k, k), (1 - accuracy) / (k - 1))
    np.fill_diagonal(like, accuracy)
    cond = _marginal(Q, probs, (axis,))
    base = float((cond * p).sum(axis=1).min())
    value = 0.0
    for s in range(k):
        joint = like[s] * p  # P(signal s, state)
        value += (cond * joint).sum(axis=1).min()
    return base - float(value)


def central_plan(Q: np.ndarray, central: tuple[int, ...]) -> int:
    """Best plan if the central scenario is taken as the forecast."""
    return int(np.argmin(Q[(slice(None), *central)]))


def vss(Q, probs, central: tuple[int, ...]) -> float:
    """Value of the stochastic solution against planning on the central forecast."""
    e = expected(Q, probs)
    return float(e[central_plan(Q, central)] - e.min())


def regret(Q) -> np.ndarray:
    """Regret of each plan by scenario."""
    return Q - Q.min(axis=0, keepdims=True)
