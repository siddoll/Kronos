import numpy as np
import pandas as pd
import pytest
from hub.config import SIGNAL_NAMES

def make_panel(seed, planted=False, strength=0.04, noise=0.02):
    """Synthetic long panel. If planted, feature 'rvol' is correlated with a
    market+sector-neutral component of fwd_ret with magnitude `strength`;
    otherwise all features are pure noise. `noise` is the idiosyncratic sigma."""
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
            alpha = strength * feats["rvol"] if planted else 0.0
            fwd = mkt + sec_shock[sectors[t]] + alpha + rng.normal(0, noise)
            rows.append({"date": d, "ticker": t, "sector": sectors[t],
                         **feats, "fwd_ret": fwd})
    return pd.DataFrame(rows)

_make_panel = make_panel  # backwards-compatible alias

@pytest.fixture
def panel_factory():
    return make_panel

@pytest.fixture
def planted_panel():
    return make_panel(0, planted=True)

@pytest.fixture
def noise_panel():
    return make_panel(1, planted=False)
