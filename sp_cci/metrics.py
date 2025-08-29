import numpy as np
from typing import Tuple

def coverage_and_width(y_true: np.ndarray, intervals: np.ndarray) -> Tuple[float, float]:
    """Compute marginal coverage and average width.
    intervals: shape (n, 2) with [lo, hi] per row
    """
    lo = intervals[:, 0]
    hi = intervals[:, 1]
    covered = (y_true >= lo) & (y_true <= hi)
    coverage = float(np.mean(covered))
    width = float(np.mean(hi - lo))
    return coverage, width
