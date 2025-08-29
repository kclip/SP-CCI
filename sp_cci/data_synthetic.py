import numpy as np
from typing import Tuple

def generate_synthetic(n: int, rho: float = 0.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lei–Candès-style synthetic data.
    Returns: X (n, d), T (n,), Y0 (n,), Y1 (n,)
    """
    rng = np.random.default_rng(seed)
    d = 10
    # Equicorrelated Gaussian
    Sigma = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
    Xp = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    # squash to [0,1]
    from scipy.stats import norm
    X = norm.cdf(Xp)

    # Propensity e(x) = 0.4 * BetaCDF_{2,4}(x1)
    from scipy.stats import beta
    e = 0.4 * beta.cdf(X[:, 0], a=2, b=4)
    T = rng.binomial(1, p=e)

    # Outcomes
    Y0 = np.zeros(n)
    def f(x): return 2.0 / (1.0 + np.exp(-12*(x - 0.5)))
    eps = rng.normal(0, 1, size=n)
    Y1 = f(X[:, 0]) * f(X[:, 1]) + eps
    return X, T, Y0, Y1

def split_indices(n: int, ratios=(0.3, 0.3, 0.2, 0.2), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n1 = int(ratios[0] * n)
    n2 = int(ratios[1] * n)
    n3 = int(ratios[2] * n)
    D_q = idx[:n1]
    D_p = idx[n1:n1+n2]
    D_cal = idx[n1+n2:n1+n2+n3]
    D_te = idx[n1+n2+n3:]
    return D_q, D_p, D_cal, D_te
