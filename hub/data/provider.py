from typing import Protocol
import pandas as pd
from .cache import OHLCVCache

class DataProvider(Protocol):
    def get_ohlcv(self, symbol: str, lookback_days: int) -> pd.DataFrame: ...
    def get_news(self, symbol: str, limit: int = 5) -> list[dict]: ...
    def get_fundamentals(self, symbol: str) -> dict: ...

class CachedProvider:
    def __init__(self, inner: DataProvider, cache: OHLCVCache):
        self.inner, self.cache = inner, cache
    def get_ohlcv(self, symbol, lookback_days):
        hit = self.cache.get(symbol)
        if hit is not None:
            return hit
        df = self.inner.get_ohlcv(symbol, lookback_days)
        self.cache.put(symbol, df)
        return df
    def get_news(self, symbol, limit=5): return self.inner.get_news(symbol, limit)
    def get_fundamentals(self, symbol): return self.inner.get_fundamentals(symbol)

class YFinanceProvider:
    def get_ohlcv(self, symbol, lookback_days):
        import yfinance as yf
        h = yf.Ticker(symbol).history(period=f"{lookback_days}d", interval="1d")
        h = h.rename(columns=str.lower)[["open","high","low","close","volume"]].astype(float)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        return h.dropna()
    def get_news(self, symbol, limit=5):
        import yfinance as yf
        out = []
        for n in (yf.Ticker(symbol).news or [])[:limit]:
            c = n.get("content", n)
            out.append({"date": str(c.get("pubDate","")), "title": c.get("title",""),
                        "source": (c.get("provider") or {}).get("displayName","")})
        return out
    def get_fundamentals(self, symbol):
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        return {"market_cap": info.get("marketCap"),
                "next_earnings_date": str(info.get("earningsTimestamp","")) or None,
                "float_shares": info.get("floatShares")}

class OpenBBProvider:  # used only when openbb is installed
    def get_ohlcv(self, symbol, lookback_days):
        from openbb import obb
        import datetime as _dt  # noqa
        data = obb.equity.price.historical(symbol, provider="yfinance")
        df = data.to_dataframe().rename(columns=str.lower)
        df = df[["open","high","low","close","volume"]].astype(float)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna().tail(lookback_days)
    def get_news(self, symbol, limit=5):
        return YFinanceProvider().get_news(symbol, limit)
    def get_fundamentals(self, symbol):
        return YFinanceProvider().get_fundamentals(symbol)

def get_default_provider(cache_dir: str) -> DataProvider:
    try:
        import openbb  # noqa
        inner: DataProvider = OpenBBProvider()
    except Exception:
        inner = YFinanceProvider()
    return CachedProvider(inner, OHLCVCache(cache_dir))
