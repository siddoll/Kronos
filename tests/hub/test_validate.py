from hub.config import HubConfig
from hub.validate import backtest_screen

def test_backtest_returns_metrics(make_df):
    import numpy as np
    frames = {
        "UP": make_df(list(np.linspace(50, 120, 200))),   # strong uptrend
        "FLAT": make_df([100.0] * 200),                    # flat
    }
    cfg = HubConfig(top_k=1, lookback_days=120)
    m = backtest_screen(frames, cfg, horizon=10, step=20)
    assert m["n"] > 0
    assert set(m) >= {"n","topk_fwd_return","universe_fwd_return","edge","hit_rate"}

def test_backtest_empty_frames_returns_zero():
    # all fetches failed -> must not crash (StopIteration guard)
    m = backtest_screen({}, HubConfig.default(), horizon=10)
    assert m["n"] == 0 and m["edge"] == 0.0

def test_backtest_picks_the_winner(make_df):
    # top-1 screen on a clear uptrend vs flat: top-k forward return should beat
    # the universe average (the rising name keeps rising in this synthetic series).
    import numpy as np
    frames = {
        "UP": make_df(list(np.linspace(50, 120, 200))),
        "FLAT": make_df([100.0] * 200),
    }
    cfg = HubConfig(top_k=1, lookback_days=120)
    m = backtest_screen(frames, cfg, horizon=10, step=20)
    assert m["topk_fwd_return"] >= m["universe_fwd_return"]
