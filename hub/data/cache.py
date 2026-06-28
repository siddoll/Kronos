import os
import time
import pandas as pd

class OHLCVCache:
    def __init__(self, cache_dir: str, ttl_hours: float = 12.0):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}.parquet")

    def get(self, symbol: str):
        path = self._path(symbol)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > self.ttl_seconds:
            return None
        return pd.read_parquet(path)

    def put(self, symbol: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(symbol))
