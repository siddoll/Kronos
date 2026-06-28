import numpy as np
import pandas as pd
from alpha.ic import rank_ic_series, ic_summary

def test_monotonic_relationship_ic_near_one():
    dates = np.repeat(pd.date_range("2022-01-01", periods=5), 10)
    x = np.tile(np.arange(10), 5).astype(float)
    y = x.copy()  # perfectly monotonic each date
    ic = rank_ic_series(dates, x, y)
    assert ic.mean() > 0.99
    s = ic_summary(ic)
    assert s["n"] == 5 and s["t_stat"] > 5

def test_planted_panel_has_positive_ic(planted_panel):
    from alpha.neutralize import neutralize
    resid = neutralize(planted_panel).values
    ic = rank_ic_series(planted_panel["date"].values, planted_panel["rvol"].values, resid)
    assert ic_summary(ic)["mean_ic"] > 0.05  # planted signal detectable

def test_noise_panel_ic_insignificant(noise_panel):
    from alpha.neutralize import neutralize
    resid = neutralize(noise_panel).values
    ic = rank_ic_series(noise_panel["date"].values, noise_panel["rvol"].values, resid)
    assert abs(ic_summary(ic)["t_stat"]) < 2.5  # not significant
