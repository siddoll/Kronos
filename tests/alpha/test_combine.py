import numpy as np
from alpha.combine import composite_score, linear_oos, lgbm_oos
from alpha.neutralize import neutralize
from alpha.splits import purged_walk_forward
from alpha.ic import rank_ic_series, ic_summary
from hub.config import SIGNAL_NAMES

def test_composite_is_per_row(planted_panel):
    s = composite_score(planted_panel, {n: 1.0 for n in SIGNAL_NAMES})
    assert len(s) == len(planted_panel)

def test_linear_oos_recovers_planted_signal(planted_panel):
    resid = neutralize(planted_panel).values
    folds = purged_walk_forward(planted_panel["date"].values, n_folds=3, purge=1)
    pred = linear_oos(planted_panel, resid, folds, list(SIGNAL_NAMES))
    mask = ~np.isnan(pred)
    ic = rank_ic_series(planted_panel["date"].values[mask], pred[mask], resid[mask])
    assert ic_summary(ic)["mean_ic"] > 0.05  # learns the rvol->target relation OOS

def test_lgbm_oos_runs(noise_panel):
    resid = neutralize(noise_panel).values
    folds = purged_walk_forward(noise_panel["date"].values, n_folds=3, purge=1)
    pred = lgbm_oos(noise_panel, resid, folds, list(SIGNAL_NAMES))
    assert pred.shape[0] == len(noise_panel)
    assert np.isnan(pred).any()  # rows before first test fold are unpredicted
