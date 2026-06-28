import os
import json
import time


class KVCache:
    def __init__(self, cache_dir: str, ttl_hours: float = 24.0):
        self.dir = cache_dir
        self.ttl = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(key))
        return os.path.join(self.dir, safe + ".json")

    def get(self, key):
        p = self._path(key)
        if not os.path.exists(p) or time.time() - os.path.getmtime(p) > self.ttl:
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def put(self, key, value) -> None:
        try:
            with open(self._path(key), "w") as f:
                json.dump(value, f)
        except Exception:
            pass
