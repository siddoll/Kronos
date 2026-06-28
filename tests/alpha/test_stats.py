import numpy as np
from alpha.stats import prob_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio

def test_psr_increases_with_sharpe():
    assert prob_sharpe_ratio(0.3, 200) > prob_sharpe_ratio(0.05, 200)

def test_psr_normal_reduces_to_phi():
    # skew=0, kurt=3, benchmark=0 -> Phi(sr*sqrt(T-1))
    from scipy.stats import norm
    val = prob_sharpe_ratio(0.2, 101, skew=0.0, kurt=3.0, sr_benchmark=0.0)
    assert abs(val - norm.cdf(0.2 * np.sqrt(100))) < 1e-9

def test_more_trials_lowers_dsr():
    a = deflated_sharpe_ratio(0.3, 200, 0.0, 3.0, sr_variance=0.01, n_trials=5)
    b = deflated_sharpe_ratio(0.3, 200, 0.0, 3.0, sr_variance=0.01, n_trials=500)
    assert b < a
