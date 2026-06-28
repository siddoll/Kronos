import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.panel import build_panel

class TrendProvider:
    def get_ohlcv(self, symbol, lookback_days):
        n = 400
        close = np.linspace(50, 110, n)
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        return pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                             "close": close, "volume": 1e6}, index=idx)
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {}

class StubExt:
    def get_upgrades_downgrades(self, t):
        return pd.DataFrame({"date": pd.to_datetime(["2021-06-01", "2021-09-01"]),
                             "up": [1, 1], "down": [0, 0]})
    def get_earnings(self, t):
        return pd.DataFrame({"date": pd.to_datetime(["2021-05-01", "2021-08-01"]),
                             "surprise": [3.0, 2.0]})

def test_panel_without_ext_has_no_extra_cols(monkeypatch):
    import alpha.panel as pmod
    monkeypatch.setattr(pmod, "load_universe", lambda name: ["AAA", "BBB"])
    cfg = AlphaConfig(history_days=400, warmup=60, universe="x")
    panel = build_panel(TrendProvider(), cfg)
    assert "rev_mom" not in panel.columns and "pead" not in panel.columns

def test_panel_with_ext_adds_features(monkeypatch):
    import alpha.panel as pmod
    monkeypatch.setattr(pmod, "load_universe", lambda name: ["AAA", "BBB"])
    cfg = AlphaConfig(history_days=400, warmup=60, universe="x")
    panel = build_panel(TrendProvider(), cfg, ext_provider=StubExt())
    assert "rev_mom" in panel.columns and "pead" in panel.columns
    assert panel["pead"].notna().any()  # at least some dates have a prior earnings
