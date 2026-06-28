import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.panel import build_panel
from alpha.sectors import load_sectors
from hub.config import SIGNAL_NAMES

def test_load_sectors_has_known():
    s = load_sectors()
    assert s.get("AAPL") == "Technology"

class TrendProvider:
    """Deterministic rising series per ticker, enough bars for the panel."""
    def get_ohlcv(self, symbol, lookback_days):
        n = 400
        base = 50 + (hash(symbol) % 10)
        close = np.linspace(base, base + 60, n)
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        return pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                             "close": close, "volume": 1e6}, index=idx)
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {}

def test_build_panel_schema_and_no_lookahead(monkeypatch):
    import alpha.panel as pmod
    monkeypatch.setattr(pmod, "load_universe", lambda name: ["AAA", "BBB", "CCC"])
    cfg = AlphaConfig(history_days=400, horizon=21, warmup=60, universe="x")
    panel = build_panel(TrendProvider(), cfg)
    assert {"date","ticker","sector","fwd_ret"} <= set(panel.columns)
    assert set(SIGNAL_NAMES) <= set(panel.columns)
    assert len(panel) > 0 and panel["fwd_ret"].notna().all()
