import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest
from hub.config import SIGNAL_NAMES

def _panel_with_planted_orthogonal(seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=40, freq="21D")
    tickers = [f"T{i:02d}" for i in range(30)]
    sectors = {t: ["A", "B", "C"][i % 3] for i, t in enumerate(tickers)}
    rows = []
    for d in dates:
        sec = {s: rng.normal(0, 0.05) for s in set(sectors.values())}
        mkt = rng.normal(0, 0.03)
        for t in tickers:
            feats = {n: rng.normal() for n in SIGNAL_NAMES}
            rev = rng.normal()
            fwd = mkt + sec[sectors[t]] + 0.05 * rev + rng.normal(0, 0.02)  # rev_mom predicts
            rows.append({"date": d, "ticker": t, "sector": sectors[t], **feats,
                         "rev_mom": rev, "pead": rng.normal(), "fwd_ret": fwd})
    return pd.DataFrame(rows)

def test_orthogonal_feature_adds_oos_value():
    panel = _panel_with_planted_orthogonal()
    cfg = AlphaConfig(n_folds=3, purge=1, extra_features=("rev_mom", "pead"))
    r = run_backtest(panel, cfg)
    assert r["ic"]["rev_mom"]["mean_ic"] > 0.05                       # standalone signal
    assert "lgbm_oos_ext" in r["ic"]
    assert r["ic"]["lgbm_oos_ext"]["mean_ic"] > r["ic"]["lgbm_oos"]["mean_ic"]  # extras help OOS

def test_backtest_ext_nan_features_handled():
    panel = _panel_with_planted_orthogonal()
    panel.loc[panel.index[:50], "rev_mom"] = np.nan  # inject missing
    cfg = AlphaConfig(n_folds=3, purge=1, extra_features=("rev_mom", "pead"))
    r = run_backtest(panel, cfg)  # must not crash
    assert r["ic"]["lgbm_oos_ext"]["n"] > 0
