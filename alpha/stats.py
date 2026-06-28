import numpy as np
from scipy.stats import norm


def prob_sharpe_ratio(sr, T, skew=0.0, kurt=3.0, sr_benchmark=0.0) -> float:
    # Probabilistic Sharpe Ratio. `kurt` is raw (non-excess), excess = kurt - 3.
    # The denominator uses the Lo (2002) / Mertens (2002) Sharpe standard-error form
    # with EXCESS kurtosis, so a normal return stream (skew=0, kurt=3) gives denom=1
    # and PSR reduces to Phi(sr * sqrt(T-1)). (Bailey & Lopez de Prado write the same
    # quantity with raw kurtosis as (g4 - 1)/4; the two differ by ~0.001 in PSR. We use
    # the excess form for the clean normal-case identity.)
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
