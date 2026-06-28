from hub.config import HubConfig

def test_default_config_has_seven_signal_weights():
    cfg = HubConfig.default()
    assert set(cfg.weights) == {
        "rvol", "breakout", "trend", "vcp", "rsi", "range_exp", "rel_strength"
    }
    assert cfg.top_k == 25
    assert all(v > 0 for v in cfg.weights.values())
