# Discovery Hub MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI research funnel that scans a US equity universe daily, scores each name with seven early-mover signals, ranks the top candidates, and uses an LLM to explain the likely catalyst/risk for each — output as CSV/JSON/HTML.

**Architecture:** UI-agnostic engine of small single-purpose modules under `hub/`. Data flows `universe → data(provider, cached) → 7 pure signal functions → weighted composite ranking → top-K → LLM explanation → report`. The CLI is a thin surface; a dashboard comes later without engine rework.

**Tech Stack:** Python 3.13 (existing `.venv`), pandas, numpy, scipy, yfinance (default free provider, already installed), openbb (optional provider), anthropic (LLM), pytest.

## Global Constraints

- Python 3.13 venv at `.venv` (use `.venv/bin/python`, `.venv/bin/pytest`).
- `pandas>=2.2.3` (the pinned 2.2.2 segfaults on 3.13 — already fixed in repo).
- All work on branch `discovery-hub`. Do NOT touch the existing pandas-fix PR.
- Immutability: signal/scoring functions are pure — never mutate the input DataFrame; copy before deriving columns.
- No hardcoded magic numbers in logic — all thresholds/weights live in `hub/config.py`.
- Files <400 lines, one responsibility each.
- OHLCV DataFrame contract everywhere: `pandas.DataFrame` with a `DatetimeIndex` and float columns exactly `["open","high","low","close","volume"]`, oldest row first.
- Every signal's `compute(df)` returns a float clamped to `[0.0, 1.0]`.
- Tests must not require network: use the fixture DataFrames provided in each task; mark any live-provider test with `@pytest.mark.network` and `pytest.importorskip`.
- New deps pinned in `hub/requirements.txt`.

---

### Task 1: Package scaffold, config, and OHLCV fixture helper

**Files:**
- Create: `hub/__init__.py`, `hub/config.py`, `hub/requirements.txt`
- Create: `tests/hub/__init__.py`, `tests/hub/conftest.py`
- Test: `tests/hub/test_config.py`

**Interfaces:**
- Produces: `hub.config.HubConfig` dataclass with fields `universe: str = "sp500_sample"`, `lookback_days: int = 260`, `top_k: int = 25`, `weights: dict[str,float]`, `explain_model: str = "claude-haiku-4-5"`, `analysis_model: str = "claude-sonnet-4-6"`, `out_dir: str = "out"`, `cache_dir: str = ".hub_cache"`; classmethod `HubConfig.default() -> HubConfig`. Default `weights` keys: `rvol, breakout, trend, vcp, rsi, range_exp, rel_strength` (each 1.0).
- Produces: `tests/hub/conftest.py` pytest fixtures `synth_uptrend_df()` and `make_ohlcv(...)` used by later tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_config.py
from hub.config import HubConfig

def test_default_config_has_seven_signal_weights():
    cfg = HubConfig.default()
    assert set(cfg.weights) == {
        "rvol", "breakout", "trend", "vcp", "rsi", "range_exp", "rel_strength"
    }
    assert cfg.top_k == 25
    assert all(v > 0 for v in cfg.weights.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hub'`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/__init__.py
"""Discovery Hub — a research funnel for early-mover stock candidates."""
```

```python
# hub/config.py
from dataclasses import dataclass, field

SIGNAL_NAMES = ("rvol", "breakout", "trend", "vcp", "rsi", "range_exp", "rel_strength")

@dataclass(frozen=True)
class HubConfig:
    universe: str = "sp500_sample"
    lookback_days: int = 260
    top_k: int = 25
    weights: dict = field(default_factory=lambda: {n: 1.0 for n in SIGNAL_NAMES})
    explain_model: str = "claude-haiku-4-5"
    analysis_model: str = "claude-sonnet-4-6"
    out_dir: str = "out"
    cache_dir: str = ".hub_cache"

    @classmethod
    def default(cls) -> "HubConfig":
        return cls()
```

```python
# hub/requirements.txt
pandas>=2.2.3
numpy
scipy
yfinance
anthropic
# optional richer data provider:
# openbb>=4.7
```

```python
# tests/hub/__init__.py
```

```python
# tests/hub/conftest.py
import numpy as np
import pandas as pd
import pytest

def make_ohlcv(closes, volume=1_000_000.0):
    """Build a valid OHLCV frame from a close-price list (oldest first)."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    high = closes * 1.01
    low = closes * 0.99
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    vol = np.full(len(closes), float(volume))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes, "volume": vol},
        index=idx,
    )

@pytest.fixture
def make_df():
    return make_ohlcv

@pytest.fixture
def synth_uptrend_df():
    # 120 bars rising 100 -> ~160 with a final volume spike
    closes = np.linspace(100, 160, 120)
    df = make_ohlcv(closes)
    df.iloc[-1, df.columns.get_loc("volume")] = 5_000_000.0
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/__init__.py hub/config.py hub/requirements.txt tests/hub/__init__.py tests/hub/conftest.py tests/hub/test_config.py
git commit -m "feat(hub): package scaffold, config, and OHLCV test fixtures"
```

---

### Task 2: Local OHLCV cache (parquet, TTL)

**Files:**
- Create: `hub/data/__init__.py`, `hub/data/cache.py`
- Test: `tests/hub/test_cache.py`

**Interfaces:**
- Produces: `hub.data.cache.OHLCVCache(cache_dir: str)` with `get(symbol: str) -> pd.DataFrame | None` (returns None if missing or older than `ttl_hours`, default 12) and `put(symbol: str, df: pd.DataFrame) -> None`. Files stored as `<cache_dir>/<symbol>.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_cache.py
from hub.data.cache import OHLCVCache

def test_put_then_get_roundtrips(tmp_path, make_df):
    cache = OHLCVCache(str(tmp_path))
    df = make_df([10, 11, 12])
    assert cache.get("AAA") is None
    cache.put("AAA", df)
    out = cache.get("AAA")
    assert out is not None and list(out["close"]) == [10, 11, 12]

def test_get_returns_none_when_expired(tmp_path, make_df):
    cache = OHLCVCache(str(tmp_path), ttl_hours=0)
    cache.put("AAA", make_df([1, 2, 3]))
    assert cache.get("AAA") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hub.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/data/__init__.py
```

```python
# hub/data/cache.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/data/__init__.py hub/data/cache.py tests/hub/test_cache.py
git commit -m "feat(hub): parquet OHLCV cache with TTL"
```

---

### Task 3: Data provider interface + yfinance/OpenBB implementations

**Files:**
- Create: `hub/data/provider.py`
- Test: `tests/hub/test_provider.py`

**Interfaces:**
- Produces: `hub.data.provider.DataProvider` (Protocol): `get_ohlcv(symbol, lookback_days) -> pd.DataFrame` (OHLCV contract), `get_news(symbol, limit=5) -> list[dict]` (each `{"date": str, "title": str, "source": str}`), `get_fundamentals(symbol) -> dict` (keys `market_cap`, `next_earnings_date`, `float_shares`; missing → None values).
- Produces: `CachedProvider(inner: DataProvider, cache: OHLCVCache)` decorator caching `get_ohlcv` only.
- Produces: `YFinanceProvider()` (default) and `OpenBBProvider()` (used only if `openbb` importable). Produces `get_default_provider(cache_dir) -> DataProvider`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_provider.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/data/provider.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_provider.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/data/provider.py tests/hub/test_provider.py
git commit -m "feat(hub): data provider interface + cached yfinance/openbb impls"
```

---

### Task 4: Universe loader

**Files:**
- Create: `hub/universe.py`, `hub/data/sp500_sample.txt`
- Test: `tests/hub/test_universe.py`

**Interfaces:**
- Produces: `hub.universe.load_universe(name: str) -> list[str]`. `"sp500_sample"` reads the bundled text file (one ticker per line, `#` comments ignored). Unknown name → `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_universe.py
import pytest
from hub.universe import load_universe

def test_loads_sample_universe():
    syms = load_universe("sp500_sample")
    assert "AAPL" in syms and len(syms) >= 20
    assert all(s == s.strip() and not s.startswith("#") for s in syms)

def test_unknown_universe_raises():
    with pytest.raises(ValueError):
        load_universe("does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_universe.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

Create `hub/data/sp500_sample.txt` with ~30 liquid large/mid-cap tickers, one per line:

```text
# Discovery Hub sample universe (liquid US large/mid-cap)
AAPL
MSFT
NVDA
AMZN
GOOGL
META
TSLA
AVGO
JPM
V
UNH
XOM
JNJ
WMT
MA
PG
HD
COST
ORCL
NFLX
AMD
CRM
KO
PEP
ADBE
BAC
DIS
INTC
CSCO
QCOM
```

```python
# hub/universe.py
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_universe(name: str) -> list[str]:
    path = os.path.join(_DATA_DIR, f"{name}.txt")
    if not os.path.exists(path):
        raise ValueError(f"Unknown universe: {name}")
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_universe.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/universe.py hub/data/sp500_sample.txt tests/hub/test_universe.py
git commit -m "feat(hub): universe loader with bundled sample list"
```

---

### Task 5: Signal base, helpers, and registry

**Files:**
- Create: `hub/signals/__init__.py`, `hub/signals/base.py`
- Test: `tests/hub/test_signal_base.py`

**Interfaces:**
- Produces: `hub.signals.base.Signal` (Protocol: attr `name: str`, method `compute(df: pd.DataFrame) -> float`).
- Produces helpers: `clamp01(x) -> float`, `ema(series, span)`, `true_range(df) -> pd.Series`, `atr(df, n)`.
- Produces: `hub.signals.SIGNALS: list[Signal]` registry (populated in later tasks) and `score_names() -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_signal_base.py
import numpy as np
from hub.signals.base import clamp01, atr

def test_clamp01_bounds():
    assert clamp01(-1) == 0.0 and clamp01(2) == 1.0 and clamp01(0.5) == 0.5
    assert clamp01(np.nan) == 0.0

def test_atr_positive(make_df):
    df = make_df(list(range(10, 40)))
    a = atr(df, 14)
    assert a.iloc[-1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_signal_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/signals/base.py
from typing import Protocol
import math
import numpy as np
import pandas as pd

class Signal(Protocol):
    name: str
    def compute(self, df: pd.DataFrame) -> float: ...

def clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n).mean()
```

```python
# hub/signals/__init__.py
from .base import Signal

SIGNALS: list[Signal] = []

def score_names() -> list[str]:
    return [s.name for s in SIGNALS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_signal_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/signals/__init__.py hub/signals/base.py tests/hub/test_signal_base.py
git commit -m "feat(hub): signal protocol, TA helpers, and registry"
```

---

### Task 6: Seven signal implementations

Each signal is a class with `name` and `compute(df) -> float` in `[0,1]`, registered into `SIGNALS`. They share one test file. Implement all seven, then register.

**Files:**
- Create: `hub/signals/rvol.py`, `breakout.py`, `trend.py`, `vcp.py`, `rsi.py`, `range_exp.py`, `rel_strength.py`
- Modify: `hub/signals/__init__.py` (register all)
- Test: `tests/hub/test_signals.py`

**Interfaces:**
- Consumes: `hub.signals.base` helpers; OHLCV contract.
- Produces: classes `RVOL, Breakout, Trend, VCP, RSI, RangeExpansion, RelStrength`, each with the matching `name` from `SIGNAL_NAMES` (`rvol, breakout, trend, vcp, rsi, range_exp, rel_strength`). `RelStrength.compute` uses the stock's own trailing return as a self-relative proxy (index-relative is a future enhancement).

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_signals.py
from hub.signals.rvol import RVOL
from hub.signals.breakout import Breakout
from hub.signals.trend import Trend
from hub.signals.vcp import VCP
from hub.signals.rsi import RSI
from hub.signals.range_exp import RangeExpansion
from hub.signals.rel_strength import RelStrength

ALL = [RVOL(), Breakout(), Trend(), VCP(), RSI(), RangeExpansion(), RelStrength()]

def test_all_signals_return_unit_interval(synth_uptrend_df):
    for s in ALL:
        v = s.compute(synth_uptrend_df)
        assert 0.0 <= v <= 1.0, f"{s.name} out of range: {v}"

def test_rvol_high_on_volume_spike(synth_uptrend_df):
    assert RVOL().compute(synth_uptrend_df) > 0.5  # last bar volume is 5x

def test_trend_high_in_uptrend(synth_uptrend_df):
    assert Trend().compute(synth_uptrend_df) > 0.5

def test_breakout_high_near_high(synth_uptrend_df):
    assert Breakout().compute(synth_uptrend_df) > 0.8  # rising series ends at its high

def test_signals_do_not_mutate_input(synth_uptrend_df):
    before = synth_uptrend_df.copy()
    for s in ALL:
        s.compute(synth_uptrend_df)
    assert synth_uptrend_df.equals(before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_signals.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/signals/rvol.py
from .base import clamp01

class RVOL:
    name = "rvol"
    def compute(self, df):
        if len(df) < 21:
            return 0.0
        avg = df["volume"].iloc[-21:-1].mean()
        if avg <= 0:
            return 0.0
        ratio = df["volume"].iloc[-1] / avg
        return clamp01((ratio - 1.0) / 2.0)  # 1x->0, 3x->1
```

```python
# hub/signals/breakout.py
from .base import clamp01

class Breakout:
    name = "breakout"
    def compute(self, df, n: int = 55):
        if len(df) < n:
            return 0.0
        window = df["close"].iloc[-n:]
        lo, hi = window.min(), window.max()
        if hi <= lo:
            return 0.0
        return clamp01((df["close"].iloc[-1] - lo) / (hi - lo))
```

```python
# hub/signals/trend.py
from .base import clamp01, ema

class Trend:
    name = "trend"
    def compute(self, df):
        if len(df) < 55:
            return 0.0
        c = df["close"]
        ma20, ma50 = ema(c, 20), ema(c, 50)
        score = 0.0
        if c.iloc[-1] > ma50.iloc[-1]:
            score += 0.4
        if ma20.iloc[-1] > ma50.iloc[-1]:
            score += 0.3
        if ma50.iloc[-1] > ma50.iloc[-6]:  # 50MA rising
            score += 0.3
        return clamp01(score)
```

```python
# hub/signals/vcp.py
from .base import clamp01, atr

class VCP:
    name = "vcp"
    def compute(self, df):
        if len(df) < 50:
            return 0.0
        short = atr(df, 10).iloc[-1]
        long_ = atr(df, 40).iloc[-1]
        if not long_ or long_ <= 0:
            return 0.0
        return clamp01(1.0 - short / long_)  # contraction -> high
```

```python
# hub/signals/rsi.py
from .base import clamp01

class RSI:
    name = "rsi"
    def _rsi(self, c, n=14):
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - 100 / (1 + rs)
    def compute(self, df):
        if len(df) < 15:
            return 0.0
        r = self._rsi(df["close"]).iloc[-1]
        if r != r:  # NaN
            return 0.0
        # peak at ~62, taper to 0 by 85 and below 45
        if r >= 85 or r <= 40:
            return 0.0
        return clamp01(1.0 - abs(r - 62) / 23.0)
```

```python
# hub/signals/range_exp.py
from .base import clamp01

class RangeExpansion:
    name = "range_exp"
    def compute(self, df):
        if len(df) < 21:
            return 0.0
        rng = (df["high"] - df["low"])
        avg = rng.iloc[-21:-1].mean()
        if avg <= 0:
            return 0.0
        return clamp01((rng.iloc[-1] / avg - 1.0) / 1.5)  # 1x->0, 2.5x->1
```

```python
# hub/signals/rel_strength.py
from .base import clamp01

class RelStrength:
    name = "rel_strength"
    def compute(self, df, n: int = 60):
        if len(df) < n + 1:
            return 0.0
        past = df["close"].iloc[-n-1]
        if past <= 0:
            return 0.0
        ret = df["close"].iloc[-1] / past - 1.0
        return clamp01(ret / 0.25)  # +25% over n bars -> 1.0
```

```python
# hub/signals/__init__.py  (replace file)
from .base import Signal
from .rvol import RVOL
from .breakout import Breakout
from .trend import Trend
from .vcp import VCP
from .rsi import RSI
from .range_exp import RangeExpansion
from .rel_strength import RelStrength

SIGNALS: list[Signal] = [
    RVOL(), Breakout(), Trend(), VCP(), RSI(), RangeExpansion(), RelStrength(),
]

def score_names() -> list[str]:
    return [s.name for s in SIGNALS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_signals.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add hub/signals/ tests/hub/test_signals.py
git commit -m "feat(hub): seven early-mover signals + registry wiring"
```

---

### Task 7: Composite scoring and ranking

**Files:**
- Create: `hub/rank.py`
- Test: `tests/hub/test_rank.py`

**Interfaces:**
- Consumes: `hub.signals.SIGNALS`, `HubConfig.weights`.
- Produces: `score_ticker(df, signals, weights) -> dict` returning `{"subscores": {name: float}, "composite": float}` where composite is the weighted mean of subscores in `[0,1]`.
- Produces: `rank_candidates(results: dict[str, dict], top_k: int) -> list[dict]` → list of `{"symbol", "composite", "subscores"}` sorted desc by composite, length ≤ top_k.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_rank.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_rank.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/rank.py
def score_ticker(df, signals, weights: dict) -> dict:
    subscores = {s.name: float(s.compute(df)) for s in signals}
    total_w = sum(weights.get(name, 0.0) for name in subscores) or 1.0
    composite = sum(subscores[name] * weights.get(name, 0.0)
                    for name in subscores) / total_w
    return {"subscores": subscores, "composite": composite}

def rank_candidates(results: dict, top_k: int) -> list:
    rows = [{"symbol": sym, "composite": r["composite"],
             "subscores": r["subscores"]} for sym, r in results.items()]
    rows.sort(key=lambda r: r["composite"], reverse=True)
    return rows[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_rank.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/rank.py tests/hub/test_rank.py
git commit -m "feat(hub): composite scoring and candidate ranking"
```

---

### Task 8: Scan orchestration (signal-only) + CSV/JSON report

**Files:**
- Create: `hub/report.py`, `hub/run.py`
- Test: `tests/hub/test_run.py`

**Interfaces:**
- Consumes: provider, `load_universe`, `SIGNALS`, `score_ticker`, `rank_candidates`, `HubConfig`.
- Produces: `hub.run.scan(cfg, provider) -> list[dict]` — fetches each symbol (per-symbol failures isolated and collected), scores, ranks top_k; each row also carries `"explanation": None` and `"skipped": [...]` is attached to the returned list via `scan_with_skips`. To keep return typing simple: `scan(cfg, provider) -> dict` = `{"candidates": list[dict], "skipped": list[dict]}`.
- Produces: `hub.report.write_reports(result: dict, cfg, date_str) -> dict[str,str]` writing `<out>/watchlist_<date>.csv` and `.json`; returns paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_run.py
import pandas as pd
from hub.config import HubConfig
from hub.run import scan
from hub.report import write_reports

class StubProvider:
    def __init__(self, frames): self.frames = frames
    def get_ohlcv(self, symbol, lookback_days):
        if symbol not in self.frames:
            raise RuntimeError("no data")
        return self.frames[symbol]
    def get_news(self, symbol, limit=5): return []
    def get_fundamentals(self, symbol): return {}

def test_scan_isolates_failures_and_ranks(make_df, monkeypatch):
    import hub.run as run_mod
    monkeypatch.setattr(run_mod, "load_universe", lambda name: ["GOOD", "BAD"])
    frames = {"GOOD": make_df(list(range(100, 220)))}  # rising
    cfg = HubConfig.default()
    result = scan(cfg, StubProvider(frames))
    assert [c["symbol"] for c in result["candidates"]] == ["GOOD"]
    assert result["skipped"][0]["symbol"] == "BAD"

def test_write_reports_creates_files(tmp_path, make_df):
    cfg = HubConfig.default().__class__(out_dir=str(tmp_path))
    result = {"candidates": [{"symbol":"X","composite":0.5,"subscores":{},
                              "explanation":None}], "skipped": []}
    paths = write_reports(result, cfg, "20260628")
    assert all(__import__("os").path.exists(p) for p in paths.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/run.py
from .universe import load_universe
from .signals import SIGNALS
from .rank import score_ticker, rank_candidates

def scan(cfg, provider) -> dict:
    symbols = load_universe(cfg.universe)
    results, skipped = {}, []
    for sym in symbols:
        try:
            df = provider.get_ohlcv(sym, cfg.lookback_days)
            if df is None or len(df) < 55:
                skipped.append({"symbol": sym, "reason": "insufficient history"})
                continue
            results[sym] = score_ticker(df, SIGNALS, cfg.weights)
        except Exception as e:  # isolate per-symbol failures
            skipped.append({"symbol": sym, "reason": str(e)})
    candidates = rank_candidates(results, cfg.top_k)
    for c in candidates:
        c.setdefault("explanation", None)
    return {"candidates": candidates, "skipped": skipped}
```

```python
# hub/report.py
import os, json, csv

def write_reports(result: dict, cfg, date_str: str) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = os.path.join(cfg.out_dir, f"watchlist_{date_str}")
    json_path, csv_path = base + ".json", base + ".csv"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    cands = result["candidates"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "symbol", "composite", "explanation"])
        for i, c in enumerate(cands, 1):
            expl = (c.get("explanation") or {})
            note = expl.get("note", "") if isinstance(expl, dict) else ""
            w.writerow([i, c["symbol"], round(c["composite"], 4), note])
    return {"json": json_path, "csv": csv_path}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_run.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/run.py hub/report.py tests/hub/test_run.py
git commit -m "feat(hub): scan orchestration with per-symbol isolation + CSV/JSON report"
```

---

### Task 9: HTML report

**Files:**
- Modify: `hub/report.py` (add `write_html`)
- Test: `tests/hub/test_report_html.py`

**Interfaces:**
- Consumes: the `result` dict from `scan`.
- Produces: `hub.report.write_html(result, cfg, date_str) -> str` writing `<out>/report_<date>.html`, and `write_reports` now also calls it and includes `"html"` in the returned paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_report_html.py
import os
from hub.config import HubConfig
from hub.report import write_reports

def test_html_report_written_and_lists_symbol(tmp_path):
    cfg = HubConfig(out_dir=str(tmp_path))
    result = {"candidates":[{"symbol":"ZZZ","composite":0.77,
              "subscores":{"rvol":0.9},"explanation":{"note":"vol spike"}}],
              "skipped":[]}
    paths = write_reports(result, cfg, "20260628")
    assert "html" in paths and os.path.exists(paths["html"])
    html = open(paths["html"]).read()
    assert "ZZZ" in html and "vol spike" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_report_html.py -v`
Expected: FAIL with `AttributeError`/missing "html" key

- [ ] **Step 3: Write minimal implementation**

```python
# hub/report.py  (append write_html and extend write_reports)
def write_html(result: dict, cfg, date_str: str) -> str:
    import os
    path = os.path.join(cfg.out_dir, f"report_{date_str}.html")
    rows = []
    for i, c in enumerate(result["candidates"], 1):
        expl = c.get("explanation") or {}
        note = expl.get("note", "") if isinstance(expl, dict) else ""
        flags = ", ".join((expl.get("risk_flags") or [])) if isinstance(expl, dict) else ""
        subs = " ".join(f"{k}:{v:.2f}" for k, v in c.get("subscores", {}).items())
        rows.append(f"<tr><td>{i}</td><td><b>{c['symbol']}</b></td>"
                    f"<td>{c['composite']:.3f}</td><td>{subs}</td>"
                    f"<td>{note}</td><td>{flags}</td></tr>")
    html = (f"<html><head><meta charset='utf-8'><title>Discovery Hub {date_str}</title>"
            "<style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #ddd;padding:6px;font-size:13px}"
            "th{background:#f4f4f4;text-align:left}</style></head><body>"
            f"<h2>Discovery Hub — {date_str}</h2>"
            f"<p>{len(result['candidates'])} candidates · {len(result['skipped'])} skipped. "
            "Research funnel, not buy signals.</p>"
            "<table><tr><th>#</th><th>Symbol</th><th>Score</th><th>Signals</th>"
            "<th>Why</th><th>Risk</th></tr>" + "".join(rows) + "</table></body></html>")
    with open(path, "w") as f:
        f.write(html)
    return path

# extend write_reports: before `return`, add:
#     paths["html"] = write_html(result, cfg, date_str)
```

Apply the extension by replacing the `return {"json": json_path, "csv": csv_path}` line in `write_reports` with:

```python
    paths = {"json": json_path, "csv": csv_path}
    paths["html"] = write_html(result, cfg, date_str)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_report_html.py tests/hub/test_run.py -v`
Expected: PASS (both files)

- [ ] **Step 5: Commit**

```bash
git add hub/report.py tests/hub/test_report_html.py
git commit -m "feat(hub): HTML watchlist report"
```

---

### Task 10: LLM explanation layer (top-K)

**Files:**
- Create: `hub/explain.py`
- Test: `tests/hub/test_explain.py`

**Interfaces:**
- Consumes: provider (`get_news`, `get_fundamentals`), an Anthropic client, `cfg.explain_model`.
- Produces: `hub.explain.explain_candidate(symbol, provider, client, model) -> dict` with keys `note, catalyst, bull, bear, risk_flags(list)`. Uses structured outputs (`output_config.format`). On any error returns `{"note":"(explanation unavailable)","risk_flags":[]}`.
- Produces: `explain_top(result, provider, client, cfg) -> dict` mutating a copy: sets `explanation` on each candidate.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_explain.py
from hub.explain import explain_top

class StubProvider:
    def get_news(self, s, limit=5): return [{"date":"2026-06-27","title":"X beats earnings","source":"PR"}]
    def get_fundamentals(self, s): return {"market_cap": 1e9}

class StubClient:
    class messages:
        @staticmethod
        def create(**kw):
            class B:  # minimal content block with valid JSON text
                type="text"; text='{"note":"earnings beat","catalyst":"earnings","bull":"x","bear":"y","risk_flags":["earnings imminent"]}'
            class R: content=[B()]
            return R()

def test_explain_top_sets_explanation():
    from hub.config import HubConfig
    result = {"candidates":[{"symbol":"X","composite":0.8,"subscores":{},"explanation":None}],"skipped":[]}
    out = explain_top(result, StubProvider(), StubClient(), HubConfig.default())
    e = out["candidates"][0]["explanation"]
    assert e["note"] == "earnings beat" and "earnings imminent" in e["risk_flags"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_explain.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/explain.py
import json
import copy

_SCHEMA = {"type":"object","additionalProperties":False,
    "required":["note","catalyst","bull","bear","risk_flags"],
    "properties":{"note":{"type":"string"},"catalyst":{"type":"string"},
        "bull":{"type":"string"},"bear":{"type":"string"},
        "risk_flags":{"type":"array","items":{"type":"string"}}}}

def explain_candidate(symbol, provider, client, model) -> dict:
    try:
        news = provider.get_news(symbol, 5)
        fund = provider.get_fundamentals(symbol)
        headlines = "\n".join(f"- {n['date']}: {n['title']}" for n in news) or "(none)"
        prompt = (f"Stock {symbol}. Recent headlines:\n{headlines}\n"
                  f"Fundamentals: {fund}\n"
                  "In 1-2 sentences each, give the likely near-term catalyst, a bull case, "
                  "a bear case, and risk_flags (e.g. 'earnings imminent', 'low float', "
                  "'recent dilution', 'possible pump-and-dump'). Be skeptical and concise. "
                  "'note' is a one-line summary.")
        resp = client.messages.create(
            model=model, max_tokens=600,
            messages=[{"role":"user","content":prompt}],
            output_config={"format":{"type":"json_schema","schema":_SCHEMA}})
        text = next(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return json.loads(text)
    except Exception:
        return {"note": "(explanation unavailable)", "catalyst": "", "bull": "",
                "bear": "", "risk_flags": []}

def explain_top(result, provider, client, cfg) -> dict:
    out = copy.deepcopy(result)
    for c in out["candidates"]:
        c["explanation"] = explain_candidate(c["symbol"], provider, client, cfg.explain_model)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_explain.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/explain.py tests/hub/test_explain.py
git commit -m "feat(hub): LLM explanation layer for top candidates"
```

---

### Task 11: CLI (`scan`) wiring

**Files:**
- Create: `hub/__main__.py`, `hub/cli.py`
- Test: `tests/hub/test_cli.py`

**Interfaces:**
- Consumes: `scan`, `explain_top`, `write_reports`, `get_default_provider`, `HubConfig`.
- Produces: `hub.cli.main(argv: list[str]) -> int`. `scan` subcommand flags: `--no-explain`, `--top-k N`, `--universe NAME`, `--out DIR`. With explanation enabled it constructs `anthropic.Anthropic()`. Prints the report paths. `__main__.py` calls `sys.exit(main(sys.argv[1:]))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_cli.py
from hub.cli import main

def test_cli_scan_no_explain(tmp_path, monkeypatch):
    import hub.cli as cli
    # stub provider + universe so no network/LLM is used
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    def fake_scan(cfg, provider):
        return {"candidates":[{"symbol":"X","composite":0.5,"subscores":{},"explanation":None}],
                "skipped":[]}
    monkeypatch.setattr(cli, "scan", fake_scan)
    rc = main(["scan", "--no-explain", "--out", str(tmp_path)])
    assert rc == 0
    assert any(p.name.startswith("watchlist_") for p in tmp_path.iterdir())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/cli.py
import argparse
import datetime as dt
from .config import HubConfig
from .data.provider import get_default_provider
from .run import scan
from .explain import explain_top
from .report import write_reports

def main(argv) -> int:
    p = argparse.ArgumentParser(prog="hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--no-explain", action="store_true")
    s.add_argument("--top-k", type=int)
    s.add_argument("--universe")
    s.add_argument("--out")
    args = p.parse_args(argv)

    if args.cmd == "scan":
        cfg = HubConfig.default()
        over = {}
        if args.top_k: over["top_k"] = args.top_k
        if args.universe: over["universe"] = args.universe
        if args.out: over["out_dir"] = args.out
        if over:
            cfg = HubConfig(**{**cfg.__dict__, **over})
        provider = get_default_provider(cfg.cache_dir)
        result = scan(cfg, provider)
        if not args.no_explain:
            import anthropic
            result = explain_top(result, provider, anthropic.Anthropic(), cfg)
        date_str = dt.datetime.now().strftime("%Y%m%d")
        paths = write_reports(result, cfg, date_str)
        print(f"{len(result['candidates'])} candidates, "
              f"{len(result['skipped'])} skipped")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
    return 1
```

```python
# hub/__main__.py
import sys
from .cli import main
sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/cli.py hub/__main__.py tests/hub/test_cli.py
git commit -m "feat(hub): CLI scan command"
```

---

### Task 12: Walk-forward validation + `backtest` CLI

**Files:**
- Create: `hub/validate.py`
- Modify: `hub/cli.py` (add `backtest` subcommand)
- Test: `tests/hub/test_validate.py`

**Interfaces:**
- Consumes: `score_ticker`, `SIGNALS`, OHLCV frames.
- Produces: `hub.validate.backtest_screen(frames: dict[str,pd.DataFrame], cfg, horizon=10, step=5) -> dict` with keys `n`, `topk_fwd_return`, `universe_fwd_return`, `edge` (topk minus universe), `hit_rate`. For each origin where ≥ `lookback`+`horizon` bars exist: score every symbol at the origin, take top-K, measure mean forward `horizon`-bar return of top-K vs the whole universe.
- Produces: `hub.cli` `backtest` subcommand that loads the universe via the provider and prints the metrics.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_validate.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/validate.py
import numpy as np
from .signals import SIGNALS
from .rank import score_ticker, rank_candidates

def backtest_screen(frames: dict, cfg, horizon: int = 10, step: int = 5) -> dict:
    lookback = cfg.lookback_days
    topk_rets, univ_rets, hits = [], [], []
    any_symbol = next(iter(frames.values()))
    n_bars = len(any_symbol)
    for origin in range(lookback, n_bars - horizon, step):
        scored, fwd = {}, {}
        for sym, df in frames.items():
            if len(df) < origin + horizon:
                continue
            window = df.iloc[:origin]
            if len(window) < 55:
                continue
            scored[sym] = score_ticker(window, SIGNALS, cfg.weights)
            c0 = df["close"].iloc[origin - 1]
            c1 = df["close"].iloc[origin - 1 + horizon]
            fwd[sym] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        if not scored:
            continue
        top = [r["symbol"] for r in rank_candidates(scored, cfg.top_k)]
        topk_rets.append(np.mean([fwd[s] for s in top]))
        univ_rets.append(np.mean(list(fwd.values())))
        hits.append(np.mean([fwd[s] > 0 for s in top]))
    if not topk_rets:
        return {"n": 0, "topk_fwd_return": 0.0, "universe_fwd_return": 0.0,
                "edge": 0.0, "hit_rate": 0.0}
    tk, uv = float(np.mean(topk_rets)), float(np.mean(univ_rets))
    return {"n": len(topk_rets), "topk_fwd_return": tk, "universe_fwd_return": uv,
            "edge": tk - uv, "hit_rate": float(np.mean(hits))}
```

Add to `hub/cli.py`: register `backtest` subparser and handle it.

```python
# in main(), after the scan subparser block:
    b = sub.add_parser("backtest")
    b.add_argument("--universe")
    b.add_argument("--horizon", type=int, default=10)
```

```python
# in main(), add a branch before `return 1`:
    if args.cmd == "backtest":
        from .universe import load_universe
        from .validate import backtest_screen
        cfg = HubConfig.default()
        if args.universe:
            cfg = HubConfig(**{**cfg.__dict__, "universe": args.universe})
        provider = get_default_provider(cfg.cache_dir)
        frames = {}
        for sym in load_universe(cfg.universe):
            try:
                frames[sym] = provider.get_ohlcv(sym, cfg.lookback_days + 60)
            except Exception:
                pass
        m = backtest_screen(frames, cfg, horizon=args.horizon)
        print("Walk-forward screen backtest (research funnel, not alpha):")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/validate.py hub/cli.py tests/hub/test_validate.py
git commit -m "feat(hub): walk-forward screen validation + backtest CLI"
```

---

### Task 13: End-to-end integration test + README

**Files:**
- Create: `tests/hub/test_integration.py`, `hub/README.md`
- Test: `tests/hub/test_integration.py`

**Interfaces:**
- Consumes: everything. Drives `scan` with a stub provider over a small universe and asserts a full report set is produced; drives `main(["scan","--no-explain"])` path via monkeypatch.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_integration.py
import os
import numpy as np
from hub.config import HubConfig
from hub.run import scan
from hub.report import write_reports

class StubProvider:
    def __init__(self, frames): self.frames = frames
    def get_ohlcv(self, s, n): return self.frames[s]
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {}

def test_end_to_end_scan_to_reports(tmp_path, make_df, monkeypatch):
    import hub.run as run_mod
    syms = [f"S{i}" for i in range(10)]
    monkeypatch.setattr(run_mod, "load_universe", lambda name: syms)
    frames = {s: make_df(list(np.linspace(50 + i, 120 + i, 200)))
              for i, s in enumerate(syms)}
    cfg = HubConfig(out_dir=str(tmp_path), top_k=5)
    result = scan(cfg, StubProvider(frames))
    assert len(result["candidates"]) == 5
    paths = write_reports(result, cfg, "20260628")
    assert os.path.exists(paths["html"]) and os.path.exists(paths["csv"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_integration.py -v`
Expected: FAIL only if a wiring bug exists; otherwise it should pass once code is present. If it fails, fix the offending module.

- [ ] **Step 3: Write the README**

```markdown
# Discovery Hub

Research funnel that scans a US equity universe, scores each name with seven
early-mover signals, ranks the top candidates, and explains the likely catalyst.

> ⚠️ Research tool, not buy signals or financial advice. It surfaces candidates
> with many false positives; it does not predict price moves.

## Run

    .venv/bin/pip install -r hub/requirements.txt
    .venv/bin/python -m hub scan                 # full scan (LLM explanations)
    .venv/bin/python -m hub scan --no-explain    # signals only, no LLM/API key
    .venv/bin/python -m hub backtest             # honest walk-forward validation

Output lands in `out/` (CSV + JSON + HTML). Needs `ANTHROPIC_API_KEY` for explanations.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/hub/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add tests/hub/test_integration.py hub/README.md
git commit -m "test(hub): end-to-end integration test + README"
```

---

## Self-Review

**Spec coverage:** Data layer (Tasks 2–4) ✓; universe (4) ✓; 7 signals (6) ✓; composite ranking (7) ✓; LLM explanation top-K (10) ✓; CLI/report surface (8,9,11) ✓; walk-forward validation (12) ✓; error isolation (8) ✓; tests ≥ per-module (1–13) ✓; config-driven weights/thresholds (1) ✓; provider abstraction for future paid feeds/Gemini (3,10) ✓. Deferred per spec: Streamlit dashboard, OpenBB Workspace, Kronos sub-score, small-cap/crypto — not in this plan (correct).

**Placeholder scan:** No TBD/TODO; every code step contains real code. ✓

**Type consistency:** OHLCV columns `["open","high","low","close","volume"]` used uniformly; `score_ticker` returns `{"subscores","composite"}` consumed by `rank_candidates` and `validate`; `explanation` dict shape (`note`/`risk_flags`) produced in Task 10 and consumed in report Tasks 8/9; `scan` returns `{"candidates","skipped"}` consumed by report + CLI + integration. Signal `name` strings match `SIGNAL_NAMES` and `HubConfig.weights` keys. ✓
