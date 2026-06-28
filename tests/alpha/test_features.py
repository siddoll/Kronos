import numpy as np
import pandas as pd
from alpha.features.revisions import revision_momentum
from alpha.features.earnings import earnings_drift


def _ud(dates, ups, downs):
    return pd.DataFrame({"date": pd.to_datetime(dates), "up": ups, "down": downs})


def test_revision_momentum_net_and_window():
    ud = _ud(["2025-01-05", "2025-01-20", "2025-03-15"], [1, 1, 0], [0, 0, 1])
    out = revision_momentum(ud, [pd.Timestamp("2025-02-01")], window_days=90)
    assert out[pd.Timestamp("2025-02-01")] == 2.0  # two ups in window, the March down excluded


def test_revision_momentum_no_lookahead():
    ud = _ud(["2025-06-01"], [1], [0])  # AFTER the as-of date
    out = revision_momentum(ud, [pd.Timestamp("2025-02-01")], window_days=90)
    assert np.isnan(out[pd.Timestamp("2025-02-01")])  # future revision must not leak


def test_earnings_drift_uses_last_prior_and_decays():
    earn = pd.DataFrame({"date": pd.to_datetime(["2025-01-10", "2024-10-10"]),
                         "surprise": [4.0, 1.0]})
    near = earnings_drift(earn, [pd.Timestamp("2025-01-20")], decay_days=60)[pd.Timestamp("2025-01-20")]
    far = earnings_drift(earn, [pd.Timestamp("2025-03-10")], decay_days=60)[pd.Timestamp("2025-03-10")]
    assert near > far > 0  # positive surprise, decaying with days-since


def test_earnings_drift_no_prior_is_nan():
    earn = pd.DataFrame({"date": pd.to_datetime(["2025-06-01"]), "surprise": [3.0]})
    out = earnings_drift(earn, [pd.Timestamp("2025-02-01")], decay_days=60)
    assert np.isnan(out[pd.Timestamp("2025-02-01")])
