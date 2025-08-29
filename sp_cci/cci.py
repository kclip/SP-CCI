import numpy as np
from typing import Callable, Dict

def compute_nonconformity_scores(
    X1: np.ndarray,
    y1: np.ndarray,
    q_lo_fn: Callable[[np.ndarray], np.ndarray],
    q_hi_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """CCI nonconformity score S_i = max{ q_lo(x)-y, y-q_hi(x), 0 } for treated set.
    """
    qlo = q_lo_fn(X1)
    qhi = q_hi_fn(X1)
    return np.maximum.reduce([qlo - y1, y1 - qhi, np.zeros_like(y1)])

def cci_eta(
    scores: np.ndarray,
    e_of_x1: np.ndarray,
    e_of_x_test: float,
    alpha: float,
) -> float:
    """Compute widening parameter eta via weighted quantile rule from Lei&Candès.
    scores: nonconformity scores for treated points
    e_of_x1: propensity e(X) for each treated point
    e_of_x_test: propensity for the test X
    """
    w = 1.0 / np.clip(e_of_x1, 1e-8, 1-1e-8)
    # Sort unique scores and scan
    uniq = np.unique(scores)
    # include a tiny grid including zero to allow eta=0
    grid = np.unique(np.concatenate([uniq, np.array([0.0])]))
    grid.sort()
    num = np.cumsum(w[np.argsort(scores)])
    den = np.sum(w) + 1.0 / np.clip(e_of_x_test, 1e-8, 1-1e-8)
    # For each threshold t, compute weighted cdf Pr[S_i <= t]
    # We'll do a straightforward scan
    best = grid[-1]
    for t in grid:
        mask = scores <= t
        if (w[mask].sum() / den) >= (1.0 - alpha):
            best = t
            break
    return float(best)

def cci_interval(q_lo: float, q_hi: float, eta: float) -> np.ndarray:
    return np.array([q_lo - eta, q_hi + eta])
