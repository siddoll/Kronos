# OpenBB Data Backbone — Implementation Plan (Research Tool v2, Step 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make OpenBB the hub's data backbone so screens/explanations use rich fundamentals + real news, not coarse yfinance — behind the existing `DataProvider` interface, with offline-testable dependency injection.

**Architecture:** Flesh out `hub/data/provider.py`'s `OpenBBProvider` (rich `get_fundamentals` + `get_news`, injectable `obb`, robust fallbacks) + a tiny JSON KV cache for dict/list results; surface key fundamentals in the report. The CLI/`explain.py`/`report.py` consume the provider interface unchanged.

**Tech Stack:** Python 3.13 (`.venv`), openbb (installed, no downgrades), pandas, pytest.

## Global Constraints

- `.venv/bin/python` / `.venv/bin/pytest`. Branch `openbb-backbone`.
- Tests must NOT hit the network: `OpenBBProvider(obb=<fake>)` injects a fake client; only a `@pytest.mark.network` smoke test (skipped by default) uses real OpenBB.
- Backward compatible: `YFinanceProvider` stays as a fallback; `get_default_provider` prefers OpenBB when importable, else yfinance. Existing `tests/hub/` stay green.
- Every OpenBB call is wrapped so a per-symbol failure returns a safe empty value (all-None dict / `[]`), never raising.
- Normalized fundamentals keys (fixed): `market_cap, pe_ratio, forward_pe, peg_ratio, earnings_growth, revenue_growth, gross_margin, net_margin, debt_to_equity, current_ratio, dividend_yield`.
- Pin `openbb` in `hub/requirements.txt`.

---

### Task 1: JSON KV cache (TTL) for dict/list results

**Files:**
- Create: `hub/data/kvcache.py`
- Test: `tests/hub/test_kvcache.py`

**Interfaces:**
- Produces `hub.data.kvcache.KVCache(cache_dir, ttl_hours=24.0)` with `get(key) -> obj|None` (None if missing/expired/corrupt) and `put(key, value)`; keys sanitized (`/`→`_`); values JSON-serializable.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_kvcache.py
from hub.data.kvcache import KVCache

def test_put_get_roundtrip(tmp_path):
    kv = KVCache(str(tmp_path))
    assert kv.get("AAPL") is None
    kv.put("AAPL", {"pe_ratio": 30.0})
    assert kv.get("AAPL") == {"pe_ratio": 30.0}

def test_ttl_expiry(tmp_path):
    kv = KVCache(str(tmp_path), ttl_hours=0)
    kv.put("X", [1, 2])
    assert kv.get("X") is None

def test_key_with_slash(tmp_path):
    kv = KVCache(str(tmp_path))
    kv.put("BRK/B", {"a": 1})
    assert kv.get("BRK/B") == {"a": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_kvcache.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/data/kvcache.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_kvcache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/data/kvcache.py tests/hub/test_kvcache.py
git commit -m "feat(hub): JSON KV cache with TTL for fundamentals/news"
```

---

### Task 2: OpenBBProvider rich `get_fundamentals` (injectable, normalized, cached)

**Files:**
- Modify: `hub/data/provider.py`, `hub/requirements.txt`
- Test: `tests/hub/test_openbb_provider.py`

**Interfaces:**
- `OpenBBProvider(obb=None, kv=None)`: `_client()` lazily imports real `obb` (setting `output_type="dataframe"`) when `obb is None`, else returns the injected client. `get_fundamentals(symbol) -> dict` normalized from `obb.equity.fundamental.metrics(symbol, provider="yfinance")` to the fixed keys (float-or-None); cached via `kv`; any error → all-None dict.
- Module helper `_to_df(x)`: returns x if a DataFrame, else `x.to_dataframe()`, else empty DataFrame.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_openbb_provider.py
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

def test_fundamentals_uses_cache(tmp_path):
    from hub.data.kvcache import KVCache
    df = pd.DataFrame([{"pe_ratio": 10.0}])
    kv = KVCache(str(tmp_path))
    p = OpenBBProvider(obb=FakeObb(df), kv=kv)
    p.get_fundamentals("AAA")
    assert kv.get("fund_AAA")["pe_ratio"] == 10.0  # written to cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_openbb_provider.py -v`
Expected: FAIL (`_FUND_KEYS`/new `OpenBBProvider` not importable as specified)

- [ ] **Step 3: Write minimal implementation**

In `hub/data/provider.py`, add the helper + constants near the top (after imports) and replace the `OpenBBProvider` class:

```python
# hub/data/provider.py — add after the existing imports
_FUND_KEYS = ["market_cap", "pe_ratio", "forward_pe", "peg_ratio", "earnings_growth",
              "revenue_growth", "gross_margin", "net_margin", "debt_to_equity",
              "current_ratio", "dividend_yield"]

def _to_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "to_dataframe"):
        return x.to_dataframe()
    return pd.DataFrame()
```

Replace the whole `class OpenBBProvider:` block with:

```python
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
```

Add `openbb` to `hub/requirements.txt` (append a line).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_openbb_provider.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/data/provider.py hub/requirements.txt tests/hub/test_openbb_provider.py
git commit -m "feat(hub): OpenBB rich normalized fundamentals (injectable + cached)"
```

---

### Task 3: OpenBB `get_news` + wire caching into the default provider

**Files:**
- Modify: `hub/data/provider.py`
- Test: `tests/hub/test_openbb_provider.py` (add cases)

**Interfaces:**
- `OpenBBProvider.get_news(symbol, limit=5) -> list[dict]` from `obb.news.company(symbol, limit=limit, provider="yfinance")`, each `{date, title, source}`; cached via `kv`; error/empty → `[]`.
- `get_default_provider(cache_dir)`: when OpenBB importable, build `CachedProvider(OpenBBProvider(kv=KVCache(cache_dir + "_kv")), OHLCVCache(cache_dir))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_openbb_provider.py  (append)
import pandas as pd
from hub.data.provider import OpenBBProvider

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_openbb_provider.py -v`
Expected: FAIL (`get_news` still delegates to yfinance → would hit network / wrong shape)

- [ ] **Step 3: Write minimal implementation**

Replace `OpenBBProvider.get_news` with:

```python
    def get_news(self, symbol, limit=5):
        if self._kv is not None:
            hit = self._kv.get(f"news_{symbol}")
            if hit is not None:
                return hit[:limit]
        out = []
        try:
            df = _to_df(self._client().news.company(symbol, limit=limit, provider="yfinance"))
            for _, r in df.iterrows():
                out.append({"date": str(r.get("date", "")),
                            "title": str(r.get("title", "")),
                            "source": str(r.get("source") or r.get("publisher") or "")})
        except Exception:
            pass
        if self._kv is not None:
            self._kv.put(f"news_{symbol}", out)
        return out[:limit]
```

Update `get_default_provider` (replace its body):

```python
def get_default_provider(cache_dir: str) -> DataProvider:
    try:
        import openbb  # noqa
        from .kvcache import KVCache
        inner: DataProvider = OpenBBProvider(kv=KVCache(cache_dir + "_kv"))
    except Exception:
        inner = YFinanceProvider()
    return CachedProvider(inner, OHLCVCache(cache_dir))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_openbb_provider.py tests/hub/test_provider.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/data/provider.py tests/hub/test_openbb_provider.py
git commit -m "feat(hub): OpenBB company news + OpenBB-backed default provider"
```

---

### Task 4: Surface fundamentals in the report + integration + live smoke

**Files:**
- Modify: `hub/explain.py`, `hub/report.py`
- Test: `tests/hub/test_fundamentals_report.py`

**Interfaces:**
- `explain_top` also sets `c["fundamentals"] = provider.get_fundamentals(c["symbol"])` on each top-K candidate (cached → cheap).
- `report.write_html` renders a "Fundamentals" cell (P/E, earnings growth, net margin) when a candidate carries `fundamentals`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_fundamentals_report.py
import os
from hub.config import HubConfig
from hub.report import write_reports

def test_html_shows_fundamentals(tmp_path):
    cfg = HubConfig(out_dir=str(tmp_path))
    result = {"candidates": [{"symbol": "AAPL", "composite": 0.7, "subscores": {},
              "explanation": {"note": "n"},
              "fundamentals": {"pe_ratio": 30.5, "earnings_growth": 0.12,
                               "net_margin": 0.25}}], "skipped": []}
    html = open(write_reports(result, cfg, "20260628")["html"]).read()
    assert "30.5" in html and "P/E" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_fundamentals_report.py -v`
Expected: FAIL (no fundamentals column)

- [ ] **Step 3: Write minimal implementation**

In `hub/explain.py`, in `explain_top`, inside the loop over `out["candidates"]`, right after the line that sets `c["explanation"] = explain_candidate(...)`, add this line (same indentation, same loop):

```python
        c["fundamentals"] = provider.get_fundamentals(c["symbol"])
```

In `hub/report.py` `write_html`, inside the per-candidate row loop, build a fundamentals cell and add it to the row. After the existing `flags = ...` line add:

```python
        fund = c.get("fundamentals") or {}
        def _f(k):
            v = fund.get(k)
            return f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        fund_cell = (f"P/E {_f('pe_ratio')} · EPSg {_f('earnings_growth')} · "
                     f"NM {_f('net_margin')}") if fund else ""
        fund_cell = _html.escape(fund_cell)
```

Add a `<th>Fundamentals</th>` header and a `<td>{fund_cell}</td>` to each row (extend the existing header row and the `rows.append(...)` f-string with the new cell).

- [ ] **Step 4: Run tests + live smoke**

Run: `.venv/bin/python -m pytest tests/hub/ -q`
Expected: PASS (all hub tests)

Live smoke (real OpenBB, confirms the backbone works end to end):

```bash
.venv/bin/python -c "
from hub.data.provider import OpenBBProvider
from hub.data.kvcache import KVCache
p = OpenBBProvider(kv=KVCache('.hub_cache_kv'))
f = p.get_fundamentals('AAPL'); n = p.get_news('AAPL', 3)
print('AAPL P/E:', f['pe_ratio'], '| earnings_growth:', f['earnings_growth'])
print('news items:', len(n), '| first:', n[0]['title'][:60] if n else None)
assert isinstance(f['pe_ratio'], float) and len(n) > 0
print('OPENBB BACKBONE OK')
"
```
Expected: prints a real P/E and a news headline, ends `OPENBB BACKBONE OK`.

- [ ] **Step 5: Commit**

```bash
git add hub/explain.py hub/report.py tests/hub/test_fundamentals_report.py
git commit -m "feat(hub): surface OpenBB fundamentals in the watchlist report"
```

---

## Self-Review

**Spec coverage:** KV cache (§3 → Task 1) ✓; rich normalized fundamentals (§4 → Task 2) ✓; OpenBB news + OpenBB-backed default provider (§4 → Task 3) ✓; fundamentals surfaced in report + LLM auto-benefits (§5 → Task 4) ✓; injectable/offline-testable (§4 → Tasks 2,3) ✓; robust per-symbol fallback (§6 → Tasks 2,3) ✓; openbb pinned (§8 → Task 2) ✓; live smoke (§7 → Task 4) ✓. Deferred per §2/§10: fundamental screening, Streamlit UI, edgartools, quantstats — not in this step (correct).

**Placeholder scan:** No TBD/TODO; every code step has real code.

**Type consistency:** `KVCache.get/put` (Task 1) used by `OpenBBProvider` (Tasks 2,3) and `get_default_provider` (Task 3). `_FUND_KEYS` + `_to_df` (Task 2) used by fundamentals + news. `get_fundamentals -> dict[_FUND_KEYS]` (Task 2) consumed by `explain_top`/report (Task 4). `get_news -> list[{date,title,source}]` (Task 3) matches the existing provider contract consumed by `explain.py`. `OpenBBProvider(obb=, kv=)` signature consistent across tasks. ✓
