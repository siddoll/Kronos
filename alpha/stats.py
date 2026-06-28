import numpy as np
from scipy.stats import norm


def prob_sharpe_ratio(sr, T, skew=0.0, kurt=3.0, sr_benchmark=0.0) -> float:
    # Bailey-Lopez de Prado PSR; kurt is raw (non-excess), so excess = kurt - 3.
    # denom uses excess kurtosis so that N(0,1) returns (skew=0, kurt=3) give denom=1
    # and PSR reduces to Phi(sr * sqrt(T-1)) in the normal case.
    excess_kurt = kurt - 3.0
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + excess_kurt / 4.0 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(max(1, T - 1)) / denom))


def expected_max_sharpe(sr_variance, n_trials) -> float:
    n = max(2, int(n_trials))
    g = 0.5772156649015329  # Euler-Mascheroni constant
    a = norm.ppf(1 - 1.0 / n)
    b = norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(max(0.0, sr_variance)) * ((1 - g) * a + g * b))


def deflated_sharpe_ratio(sr, T, skew, kurt, sr_variance, n_trials) -> float:
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    return prob_sharpe_ratio(sr, T, skew, kurt, sr0)
