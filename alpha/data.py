import os
import time
import pandas as pd


class _ParquetCache:
    def __init__(self, cache_dir, ttl_hours=72.0):
        self.dir = cache_dir
        self.ttl = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key):
        p = os.path.join(self.dir, key + ".parquet")
        if not os.path.exists(p) or time.time() - os.path.getmtime(p) > self.ttl:
            return None
        return pd.read_parquet(p)

    def put(self, key, df):
        df.to_parquet(os.path.join(self.dir, key + ".parquet"))


def _col(df, name):
    """Return df[name] as str Series, or empty-string Series if absent."""
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series([""] * len(df), index=df.index)


class ExternalDataProvider:
    def __init__(self, cache_dir=".alpha_ext_cache"):
        self.cache = _ParquetCache(cache_dir)

    @staticmethod
    def _norm_ud(raw) -> pd.DataFrame:
        cols = ["date", "up", "down"]
        if raw is None or len(raw) == 0:
            return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
        d = raw.reset_index()
        date_col = next((c for c in d.columns if "date" in str(c).lower()), d.columns[0])
        dt = pd.to_datetime(d[date_col], errors="coerce")
        try:
            dt = dt.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        act = _col(d, "Action").str.lower()
        pta = _col(d, "priceTargetAction").str.lower()
        up = ((act == "up") | (pta == "raises")).astype(int)
        down = ((act == "down") | (pta == "lowers")).astype(int)
        return pd.DataFrame({"date": dt, "up": up, "down": down}).dropna(subset=["date"]).reset_index(drop=True)

    @staticmethod
    def _norm_earn(raw) -> pd.DataFrame:
        if raw is None or len(raw) == 0:
            return pd.DataFrame({"date": pd.Series(dtype="object"), "surprise": pd.Series(dtype="float")})
        d = raw.reset_index()
        date_col = d.columns[0]
        dt = pd.to_datetime(d[date_col], errors="coerce")
        try:
            dt = dt.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        sur = pd.to_numeric(d.get("Surprise(%)"), errors="coerce")
        return pd.DataFrame({"date": dt, "surprise": sur}).dropna(subset=["date"]).reset_index(drop=True)

    def get_upgrades_downgrades(self, ticker) -> pd.DataFrame:
        hit = self.cache.get(f"ud_{ticker}")
        if hit is not None:
            return hit
        import yfinance as yf
        df = self._norm_ud(yf.Ticker(ticker).upgrades_downgrades)
        self.cache.put(f"ud_{ticker}", df)
        return df

    def get_earnings(self, ticker) -> pd.DataFrame:
        hit = self.cache.get(f"earn_{ticker}")
        if hit is not None:
            return hit
        import yfinance as yf
        try:
            raw = yf.Ticker(ticker).get_earnings_dates(limit=40)
        except Exception:
            raw = None
        df = self._norm_earn(raw)
        self.cache.put(f"earn_{ticker}", df)
        return df
