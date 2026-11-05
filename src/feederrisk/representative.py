"""Representative days: k-means on daily features, nearest real day as medoid, plus forced extremes."""

import numpy as np


def select_days(features: np.ndarray, k: int, forced: list[int], seed: int = 1, iters: int = 50):
    """Return day indices and the number of days each one stands for (weights sum to the year)."""
    x = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.array([x[lab == j].mean(axis=0) if (lab == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    medoids = sorted({int(((x - c) ** 2).sum(axis=1).argmin()) for c in centers} | set(forced))
    m = np.array(medoids)
    assign = ((x[:, None, :] - x[m][None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    weights = np.bincount(assign, minlength=len(m)).astype(float)
    # Forced extremes stand for themselves alone: their weight must not inflate rare conditions.
    forced_set = set(forced)
    others = np.array([i for i in range(len(m)) if medoids[i] not in forced_set])
    for f in forced_set:
        j = medoids.index(f)
        spare = weights[j] - 1
        if spare > 0:
            weights[j] = 1
            d = ((x[f] - x[m[others]]) ** 2).sum(axis=1)
            weights[others[d.argmin()]] += spare
    return m, weights
