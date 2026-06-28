import numpy as np
import pandas as pd
import pytest

def make_ohlcv(closes, volume=1_000_000.0):
    """Build a valid OHLCV frame from a close-price list (oldest first)."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    high = closes * 1.01
    low = closes * 0.99
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    vol = np.full(len(closes), float(volume))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes, "volume": vol},
        index=idx,
    )

@pytest.fixture
def make_df():
    return make_ohlcv

@pytest.fixture
def synth_uptrend_df():
    # 120 bars rising 100 -> ~160 with a final volume spike
    closes = np.linspace(100, 160, 120)
    df = make_ohlcv(closes)
    df.iloc[-1, df.columns.get_loc("volume")] = 5_000_000.0
    return df
