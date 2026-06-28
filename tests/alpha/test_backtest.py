from alpha.config import AlphaConfig
from alpha.backtest import run_backtest

def test_backtest_detects_planted_signal(planted_panel):
    cfg = AlphaConfig(n_folds=3, purge=1)
    r = run_backtest(planted_panel, cfg)
    assert r["ic"]["rvol"]["mean_ic"] > 0.05          # raw signal IC
    assert r["ic"]["lgbm_oos"]["n"] > 0               # OOS preds produced
    assert "dsr" in r["portfolio"]["composite"]
    assert r["n_rows"] == len(planted_panel)

def test_backtest_noise_not_significant(noise_panel):
    cfg = AlphaConfig(n_folds=3, purge=1)
    r = run_backtest(noise_panel, cfg)
    assert abs(r["ic"]["rvol"]["t_stat"]) < 3.0
