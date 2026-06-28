import numpy as np
from alpha.portfolio import long_short

def test_perfect_score_is_profitable_and_costs_reduce(planted_panel):
    # score = the realized fwd_ret (oracle) -> long winners, short losers
    score = planted_panel["fwd_ret"].values
    free = long_short(planted_panel, score, n_quantiles=5, cost_bps=0.0)
    paid = long_short(planted_panel, score, n_quantiles=5, cost_bps=50.0)
    assert free["gross"].mean() > 0
    assert paid["net"].mean() < free["net"].mean()  # costs drag

def test_random_score_centered_near_zero(noise_panel):
    rng = np.random.RandomState(3)
    score = rng.normal(size=len(noise_panel))
    r = long_short(noise_panel, score, n_quantiles=5, cost_bps=0.0)
    assert abs(r["gross"].mean()) < 0.02
