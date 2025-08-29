import numpy as np
from typing import Callable, Dict, Tuple

def _ppi_debiased_group_loss(
    # One real treated example (X_i, y_i) and r synthetic (X_tilde, y_tilde)
    X_i: np.ndarray,
    y_i: float,
    X_group: np.ndarray,
    yhat_group: np.ndarray,
    q_lo_fn: Callable[[np.ndarray], np.ndarray],
    q_hi_fn: Callable[[np.ndarray], np.ndarray],
    loss_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    w_i: float,
    w_tilde_group: np.ndarray,
    yhat_i: float,
    eta: float,
) -> float:
    # synthetic term
    qlo_g = q_lo_fn(X_group)
    qhi_g = q_hi_fn(X_group)
    l_syn = loss_fn(yhat_group, qlo_g, qhi_g, eta)  # predicted labels in group
    syn_term = float(np.mean(w_tilde_group * l_syn))
    # bias correction on the real point
    qlo_i = q_lo_fn(X_i[None, :])[0]
    qhi_i = q_hi_fn(X_i[None, :])[0]
    l_pred = loss_fn(np.array([yhat_i]), np.array([qlo_i]), np.array([qhi_i]), eta)[0]
    l_true = loss_fn(np.array([y_i]), np.array([qlo_i]), np.array([qhi_i]), eta)[0]
    correction = w_i * (l_pred - l_true)
    return syn_term - correction

def _binary_miscoverage(y: np.ndarray, qlo: np.ndarray, qhi: np.ndarray, eta: float) -> np.ndarray:
    lo = qlo - eta
    hi = qhi + eta
    return ( (y < lo) | (y > hi) ).astype(float)

def spcci_eta(
    # Real treated calibration
    X1: np.ndarray,
    y1: np.ndarray,
    # Synthetic groups: list of arrays aligned with y1
    groups_X: list,
    groups_yhat: list,
    # Weight functions/values
    w_real: np.ndarray,
    w_syn: list,   # list of arrays per group
    # Quantile models
    q_lo_fn: Callable[[np.ndarray], np.ndarray],
    q_hi_fn: Callable[[np.ndarray], np.ndarray],
    # Synthetic generator for correction term on real point (sample per real point)
    yhat_i: np.ndarray,
    # RCPS params
    alpha: float,
    delta: float,
    grid: np.ndarray = None,
) -> float:
    """Compute eta for SP-CCI by minimizing UB <= alpha with Hoeffding-style bound.
    Returns eta.
    """
    n1 = len(y1)
    if grid is None:
        # Build a small grid from empirical residual magnitudes
        # You may refine this grid depending on your models
        grid = np.linspace(0.0, np.percentile(np.abs(y1 - np.median(y1)), 95), 101)

    def Lhat(eta: float) -> float:
        vals = []
        for i in range(n1):
            vals.append(
                _ppi_debiased_group_loss(
                    X1[i], float(y1[i]),
                    groups_X[i], groups_yhat[i],
                    q_lo_fn, q_hi_fn,
                    _binary_miscoverage,
                    float(w_real[i]),
                    w_syn[i],
                    float(yhat_i[i]),
                    eta,
                )
            )
        return float(np.mean(vals))

    ub = lambda eta: Lhat(eta) + (np.sqrt(np.log(1.0/delta) / (2.0 * max(1, n1))))

    best = grid[-1]
    for t in grid:
        if ub(float(t)) <= alpha:
            best = float(t)
            break
    return best

def spcci_interval(q_lo: float, q_hi: float, eta: float) -> np.ndarray:
    return np.array([q_lo - eta, q_hi + eta])
