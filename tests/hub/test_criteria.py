import numpy as np
import pandas as pd
from hub.screen.criteria import (near_52w_high, momentum_12_1_positive, above_sma,
                                  rsi_between, macd_bullish, adx_above, CritResult)

def _price(closes):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": 1e6}, index=idx)

def test_near_high_passes_at_top_fails_after_drop():
    rising = _price(np.linspace(100, 160, 300))
    r = near_52w_high(0.05).evaluate(rising, None)
    assert r.passed and 0 <= r.score <= 1
    dropped = _price(list(np.linspace(100, 160, 280)) + list(np.linspace(160, 120, 20)))
    assert not near_52w_high(0.05).evaluate(dropped, None).passed

def test_momentum_12_1_sign():
    up = _price(np.linspace(50, 150, 300))
    assert momentum_12_1_positive().evaluate(up, None).passed
    down = _price(np.linspace(150, 60, 300))
    assert not momentum_12_1_positive().evaluate(down, None).passed

def test_above_sma_and_rsi_and_macd_run():
    up = _price(np.linspace(80, 160, 300))
    assert above_sma(200).evaluate(up, None).passed
    for c in (rsi_between(0, 100), macd_bullish(), adx_above(0)):
        res = c.evaluate(up, None)
        assert isinstance(res, CritResult) and 0 <= res.score <= 1

def test_insufficient_history_fails_safe():
    short = _price([100, 101, 102])
    r = momentum_12_1_positive().evaluate(short, None)
    assert not r.passed and r.score == 0.0  # no crash

def test_near_high_needs_full_year():
    # < window bars must NOT spuriously pass (review I1): a 60-bar rising name is
    # not a 52-week high.
    short = _price(np.linspace(100, 160, 60))
    r = near_52w_high(0.05, window=252).evaluate(short, None)
    assert not r.passed and r.score == 0.0
