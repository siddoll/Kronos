import numpy as np
import pandas as pd
from hub.screen.criteria import rvol_above, short_momentum_positive
from hub.screen.library import get_preset

def _price(closes, vols=None):
    closes = np.asarray(closes, float)
    n = len(closes)
    vols = np.full(n, 1e6) if vols is None else np.asarray(vols, float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": vols}, index=idx)

def test_rvol_spike():
    vols = [1e6]*40 + [3e6]   # last bar 3x
    r = rvol_above(1.5).evaluate(_price(np.linspace(100, 110, 41), vols), None)
    assert r.passed and r.value > 2.5 and 0 <= r.score <= 1

def test_short_momentum_sign():
    up = _price(np.linspace(50, 80, 60))
    assert short_momentum_positive(20).evaluate(up, None).passed
    down = _price(np.linspace(80, 50, 60))
    assert not short_momentum_positive(20).evaluate(down, None).passed

def test_short_history_failsafe():
    r = short_momentum_positive(20).evaluate(_price([100, 101, 102]), None)
    assert not r.passed and r.score == 0.0

def test_preset_exists():
    crits = get_preset("momentum_catalyst")
    assert any(c.name == "rvol_above" for c in crits)
    assert any(c.name == "short_momentum" for c in crits)
