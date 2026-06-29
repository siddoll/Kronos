import numpy as np
import pandas as pd
from hub.screen.forward_test import forward_test
from hub.screen.criteria import short_momentum_positive, above_sma

def _frame(closes):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2022-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": 1e6}, index=idx)

CRIT = [short_momentum_positive(20), above_sma(50)]

def test_trending_name_is_picked_and_up():
    frames = {"UP": _frame(np.linspace(50, 150, 300)), "FLAT": _frame([100.0]*300)}
    r = forward_test(frames, CRIT, horizons=(5, 10), step=10, warmup=60)
    assert r["n_dates"] > 0
    assert set(r["horizons"][5]) == {"hit_rate", "pick_return", "market_return", "edge", "n"}
    assert r["horizons"][5]["hit_rate"] > 0.6     # the persistent uptrend keeps rising

def test_random_walk_has_no_edge():
    # HONESTY GUARD: noise must yield ~coin-flip hit-rate and ~0 edge — the harness
    # must not manufacture an edge.
    rng = np.random.RandomState(0)
    frames = {}
    for i in range(20):
        steps = rng.normal(0, 0.01, 320)
        frames[f"T{i}"] = _frame(100 * np.exp(np.cumsum(steps)))
    r = forward_test(frames, CRIT, horizons=(5, 10), step=5, warmup=60)
    assert 0.40 < r["horizons"][5]["hit_rate"] < 0.60
    assert abs(r["horizons"][5]["edge"]) < 0.03

def test_empty_is_safe():
    r = forward_test({}, CRIT, horizons=(5,))
    assert r["n_dates"] == 0 and r["horizons"][5]["n"] == 0

def test_nan_forward_close_does_not_poison_aggregates():
    # HONESTY GUARD: if a forward close is NaN (mid-series gap), skip that return
    # so NaN does NOT poison pick_return, market_return, edge via np.mean()
    closes_good = list(np.linspace(100, 110, 300))
    closes_bad = list(np.linspace(100, 110, 300))
    # Plant NaN in the forward window for some dates
    # origin=260, h=5 → j=264. Plant NaN at index 264.
    closes_bad[264] = np.nan

    frames = {"GOOD": _frame(closes_good), "BAD": _frame(closes_bad)}
    r = forward_test(frames, CRIT, horizons=(5,), step=1, warmup=60)

    # All aggregates must be finite (not NaN) — even with one name having a NaN in forward window
    h = 5
    for key in ["pick_return", "market_return", "edge"]:
        val = r["horizons"][h][key]
        assert val == val, f"horizon {h} {key} is NaN: {val}"
