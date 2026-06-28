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

def test_noise_has_no_systematic_ic(panel_factory):
    from alpha.neutralize import neutralize
    # A single noise draw can exceed |t|=2 ~5% of the time (that IS the false-positive
    # rate). The honest guarantee is that noise has NO systematic edge: averaged over
    # many seeds, mean |t| stays near the null expectation (~0.8), well below 2.
    ts = []
    for seed in range(12):
        p = panel_factory(seed + 100, planted=False)
        resid = neutralize(p).values
        ic = rank_ic_series(p["date"].values, p["rvol"].values, resid)
        ts.append(abs(ic_summary(ic)["t_stat"]))
    assert np.mean(ts) < 1.5  # no systematic significance across draws

def test_modest_signal_is_detectable(panel_factory):
    # I3: a weaker planted signal (lower SNR) must still be detected with the right
    # sign — proves sensitivity near the realistic regime, not just a giant t-stat.
    from alpha.neutralize import neutralize
    p = panel_factory(7, planted=True, strength=0.012, noise=0.05)
    resid = neutralize(p).values
    ic = rank_ic_series(p["date"].values, p["rvol"].values, resid)
    s = ic_summary(ic)
    assert s["mean_ic"] > 0.02 and s["t_stat"] > 2  # detectable but modest
