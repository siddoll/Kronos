import pandas as pd
from hub.data.provider import OpenBBProvider, _FUND_KEYS

class _FundamentalNS:
    def __init__(self, df, raise_it=False):
        self._df, self._raise = df, raise_it
    def metrics(self, symbol, provider=None):
        if self._raise:
            raise RuntimeError("api down")
        return self._df

class _EquityNS:
    def __init__(self, df, raise_it=False):
        self.fundamental = _FundamentalNS(df, raise_it)

class FakeObb:
    """Minimal obb stand-in: obb.equity.fundamental.metrics(...)."""
    def __init__(self, metrics_df=None, raise_it=False):
        self.equity = _EquityNS(metrics_df, raise_it)

def test_fundamentals_normalized():
    df = pd.DataFrame([{"pe_ratio": 30.5, "earnings_growth": 0.12, "market_cap": 3.1e12}])
    out = OpenBBProvider(obb=FakeObb(df)).get_fundamentals("AAPL")
    assert set(out) == set(_FUND_KEYS)
    assert out["pe_ratio"] == 30.5 and out["earnings_growth"] == 0.12
    assert out["forward_pe"] is None  # absent column -> None

def test_fundamentals_error_is_all_none():
    out = OpenBBProvider(obb=FakeObb(raise_it=True)).get_fundamentals("AAPL")
    assert all(v is None for v in out.values()) and set(out) == set(_FUND_KEYS)

def test_one_bad_metric_does_not_drop_others():
    # OpenBB free providers occasionally emit string sentinels; a non-numeric value
    # for one key must NOT wipe out the valid metrics (review finding #1).
    df = pd.DataFrame([{"market_cap": "N/A", "pe_ratio": 30.5, "net_margin": 0.25}])
    out = OpenBBProvider(obb=FakeObb(df)).get_fundamentals("AAPL")
    assert out["market_cap"] is None
    assert out["pe_ratio"] == 30.5 and out["net_margin"] == 0.25

def test_fundamentals_uses_cache(tmp_path):
    from hub.data.kvcache import KVCache
    df = pd.DataFrame([{"pe_ratio": 10.0}])
    kv = KVCache(str(tmp_path))
    p = OpenBBProvider(obb=FakeObb(df), kv=kv)
    p.get_fundamentals("AAA")
    assert kv.get("fund_AAA")["pe_ratio"] == 10.0  # written to cache


# ---- Task 3: news tests ----

class _News:
    def __init__(self, df): self._df = df
    def company(self, symbol, limit=5, provider=None): return self._df

class FakeObbNews:
    def __init__(self, news_df):
        class N: pass
        self.news = _News(news_df)

def test_news_mapped():
    df = pd.DataFrame([{"date": "2026-06-27", "title": "X beats", "source": "PR"},
                       {"date": "2026-06-26", "title": "Y", "source": "Wire"}])
    out = OpenBBProvider(obb=FakeObbNews(df)).get_news("AAPL", limit=5)
    assert out[0]["title"] == "X beats" and out[0]["source"] == "PR"

def test_news_empty():
    out = OpenBBProvider(obb=FakeObbNews(pd.DataFrame())).get_news("AAPL")
    assert out == []
