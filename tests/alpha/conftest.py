import numpy as np
import pandas as pd
import pytest
from hub.config import SIGNAL_NAMES

def _make_panel(seed, planted):
    """Synthetic long panel. If planted, feature 'rvol' is correlated with a
    market+sector-neutral component of fwd_ret; otherwise all features are noise."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=40, freq="21D")
    tickers = [f"T{i:02d}" for i in range(30)]
    sectors = {t: ["A", "B", "C"][i % 3] for i, t in enumerate(tickers)}
    rows = []
    for d in dates:
        sec_shock = {s: rng.normal(0, 0.05) for s in set(sectors.values())}
        mkt = rng.normal(0, 0.03)
        for t in tickers:
            feats = {name: rng.normal() for name in SIGNAL_NAMES}
            alpha = 0.04 * feats["rvol"] if planted else 0.0
            fwd = mkt + sec_shock[sectors[t]] + alpha + rng.normal(0, 0.02)
            rows.append({"date": d, "ticker": t, "sector": sectors[t],
                         **feats, "fwd_ret": fwd})
    return pd.DataFrame(rows)

@pytest.fixture
def planted_panel():
    return _make_panel(0, planted=True)

@pytest.fixture
def noise_panel():
    return _make_panel(1, planted=False)
