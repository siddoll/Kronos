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
