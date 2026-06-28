import numpy as np
import pandas as pd
from hub.screen.screener import run_screen
from hub.screen.criteria import above_sma
from hub.screen.library import pe_below

def _price(trend):
    n = 300
    closes = np.linspace(100, 100 + trend, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": 1e6}, index=idx)

class StubProvider:
    def __init__(self):
        self.frames = {"UP": _price(+60), "DOWN": _price(-40)}
        self.funds = {"UP": {"pe_ratio": 20.0}, "DOWN": {"pe_ratio": 20.0}}
        self.fund_calls = []
    def get_ohlcv(self, s, lookback_days): return self.frames[s]
    def get_fundamentals(self, s):
        self.fund_calls.append(s); return self.funds[s]
    def get_news(self, s, limit=5): return []

def test_technical_filter_runs_before_fundamentals():
    p = StubProvider()
    res = run_screen(["UP", "DOWN"], p, [above_sma(200), pe_below(40)], top_k=10)
    syms = [c["symbol"] for c in res["candidates"]]
    assert "UP" in syms and "DOWN" not in syms          # DOWN fails above_sma (hard)
    assert p.fund_calls == ["UP"]                        # fundamentals fetched only for survivor

def test_fundamental_hard_filter_drops():
    p = StubProvider(); p.funds["UP"] = {"pe_ratio": 99.0}
    res = run_screen(["UP"], p, [above_sma(200), pe_below(40)], top_k=10)
    assert res["candidates"] == []                       # UP passes tech but fails pe<40

def test_ranking_and_payload():
    p = StubProvider()
    res = run_screen(["UP"], p, [above_sma(200), pe_below(40)], top_k=10)
    c = res["candidates"][0]
    assert 0 <= c["composite"] <= 1 and "above_sma200" in c["criteria"]
    assert c["fundamentals"]["pe_ratio"] == 20.0 and c["explanation"] is None
