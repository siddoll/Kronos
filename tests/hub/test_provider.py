import pandas as pd
from hub.data.provider import CachedProvider
from hub.data.cache import OHLCVCache

class FakeProvider:
    def __init__(self): self.calls = 0
    def get_ohlcv(self, symbol, lookback_days):
        self.calls += 1
        return pd.DataFrame(
            {"open":[1.0],"high":[1.0],"low":[1.0],"close":[1.0],"volume":[1.0]},
            index=pd.to_datetime(["2026-01-01"]),
        )
    def get_news(self, symbol, limit=5): return []
    def get_fundamentals(self, symbol): return {"market_cap": None}

def test_cached_provider_only_fetches_once(tmp_path):
    inner = FakeProvider()
    cp = CachedProvider(inner, OHLCVCache(str(tmp_path)))
    cp.get_ohlcv("AAA", 30)
    cp.get_ohlcv("AAA", 30)
    assert inner.calls == 1  # second served from cache
