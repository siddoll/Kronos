from typing import Protocol
import pandas as pd
from .cache import OHLCVCache

_FUND_KEYS = ["market_cap", "pe_ratio", "forward_pe", "peg_ratio", "earnings_growth",
              "revenue_growth", "gross_margin", "net_margin", "debt_to_equity",
              "current_ratio", "dividend_yield"]


def _to_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "to_dataframe"):
        return x.to_dataframe()
    return pd.DataFrame()

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
        # lookback_days is a TRADING-BAR count (used as bar index in signals/backtest).
        # yfinance period="Nd" is CALENDAR days (~0.69 trading bars each), so fetch a
        # generous calendar window then tail to the requested number of bars.
        cal_days = int(lookback_days * 1.7) + 60
        h = yf.Ticker(symbol).history(period=f"{cal_days}d", interval="1d")
        h = h.rename(columns=str.lower)[["open","high","low","close","volume"]].astype(float)
        h.index = pd.to_datetime(h.index).tz_localize(None)
        return h.dropna().tail(lookback_days)
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

class OpenBBProvider:
    """Rich data via OpenBB. obb is injectable for offline testing."""
    def __init__(self, obb=None, kv=None):
        self._obb = obb
        self._kv = kv

    def _client(self):
        if self._obb is None:
            from openbb import obb
            obb.user.preferences.output_type = "dataframe"
            self._obb = obb
        return self._obb

    def get_ohlcv(self, symbol, lookback_days):
        df = _to_df(self._client().equity.price.historical(symbol, provider="yfinance"))
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna().tail(lookback_days)

    def get_fundamentals(self, symbol):
        if self._kv is not None:
            hit = self._kv.get(f"fund_{symbol}")
            if hit is not None:
                return hit
        out = {k: None for k in _FUND_KEYS}
        try:
            df = _to_df(self._client().equity.fundamental.metrics(symbol, provider="yfinance"))
            if len(df):
                row = df.iloc[-1]
                for k in _FUND_KEYS:
                    v = row.get(k) if hasattr(row, "get") else None
                    out[k] = float(v) if v is not None and v == v else None
        except Exception:
            pass
        if self._kv is not None:
            self._kv.put(f"fund_{symbol}", out)
        return out

    def get_news(self, symbol, limit=5):
        return YFinanceProvider().get_news(symbol, limit)  # upgraded in Task 3

def get_default_provider(cache_dir: str) -> DataProvider:
    try:
        import openbb  # noqa
        inner: DataProvider = OpenBBProvider()
    except Exception:
        inner = YFinanceProvider()
    return CachedProvider(inner, OHLCVCache(cache_dir))
