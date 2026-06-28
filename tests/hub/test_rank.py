from hub.signals import SIGNALS
from hub.config import HubConfig
from hub.rank import score_ticker, rank_candidates

def test_score_ticker_composite_in_unit_interval(synth_uptrend_df):
    cfg = HubConfig.default()
    out = score_ticker(synth_uptrend_df, SIGNALS, cfg.weights)
    assert set(out["subscores"]) == set(cfg.weights)
    assert 0.0 <= out["composite"] <= 1.0

def test_rank_orders_and_truncates():
    results = {
        "A": {"composite": 0.9, "subscores": {}},
        "B": {"composite": 0.1, "subscores": {}},
        "C": {"composite": 0.5, "subscores": {}},
    }
    ranked = rank_candidates(results, top_k=2)
    assert [r["symbol"] for r in ranked] == ["A", "C"]
