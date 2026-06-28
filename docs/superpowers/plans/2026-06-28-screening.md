# Configurable Screening — Implementation Plan (Research Tool v2, Step 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A stock screener — technical criteria (pandas-ta) + fundamental criteria (OpenBB backbone) combined into named preset theses — that returns ranked, criteria-annotated picks via `hub screen --preset <name>`.

**Architecture:** New `hub/screen/` package: a `Criterion` model, technical + fundamental criteria, a `library` of presets, and a **staged** `screener` (technical hard-filters first on price-only; fundamentals fetched only for survivors). Reuses the OpenBB provider, `explain.py`, and `report.py`.

**Tech Stack:** Python 3.13 (`.venv`), pandas-ta 0.4.71b0 (verified on pandas 3.0), the merged hub + OpenBB backbone.

## Global Constraints

- `.venv/bin/python` / `.venv/bin/pytest`. Branch `screening`. Add `pandas-ta` to `hub/requirements.txt`.
- Tests must NOT hit the network: synthetic price DataFrames + fundamentals dicts; stub provider for the screener.
- A criterion that errors (insufficient history, missing column) returns `CritResult(passed=False, score=0.0, value=nan)` — never raises.
- A missing fundamental (`None`) → the fundamental criterion **fails closed** (`passed=False`), so a hard fundamental filter never admits a name with unknown data.
- Staged screener: fundamentals (`provider.get_fundamentals`) are requested ONLY for names that pass all hard technical filters.
- OHLCV DataFrame contract: columns `open,high,low,close,volume`, oldest first (the hub provider already yields this).

---

### Task 1: Criterion model + technical criteria

**Files:**
- Create: `hub/screen/__init__.py`, `hub/screen/criteria.py`
- Modify: `hub/requirements.txt` (add `pandas-ta`)
- Test: `tests/hub/test_criteria.py`

**Interfaces:**
- `hub.screen.criteria.CritResult(passed: bool, score: float, value: float)` (dataclass).
- `Criterion(name, kind, hard, fn)` with `evaluate(price_df, fundamentals) -> CritResult` (wraps `fn` in try/except → fail-safe).
- Technical factories returning `Criterion(kind="technical")`: `near_52w_high(within_pct=0.05, window=252, hard=True)`, `momentum_12_1_positive(hard=True)`, `above_sma(n=200, hard=True)`, `rsi_between(lo=50, hi=70, hard=True)`, `macd_bullish(hard=True)`, `adx_above(n=25, hard=True)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_criteria.py
import numpy as np
import pandas as pd
from hub.screen.criteria import (near_52w_high, momentum_12_1_positive, above_sma,
                                  rsi_between, macd_bullish, adx_above, CritResult)

def _price(closes):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": 1e6}, index=idx)

def test_near_high_passes_at_top_fails_after_drop():
    rising = _price(np.linspace(100, 160, 300))
    r = near_52w_high(0.05).evaluate(rising, None)
    assert r.passed and 0 <= r.score <= 1
    dropped = _price(list(np.linspace(100, 160, 280)) + list(np.linspace(160, 120, 20)))
    assert not near_52w_high(0.05).evaluate(dropped, None).passed

def test_momentum_12_1_sign():
    up = _price(np.linspace(50, 150, 300))
    assert momentum_12_1_positive().evaluate(up, None).passed
    down = _price(np.linspace(150, 60, 300))
    assert not momentum_12_1_positive().evaluate(down, None).passed

def test_above_sma_and_rsi_and_macd_run():
    up = _price(np.linspace(80, 160, 300))
    assert above_sma(200).evaluate(up, None).passed
    for c in (rsi_between(0, 100), macd_bullish(), adx_above(0)):
        res = c.evaluate(up, None)
        assert isinstance(res, CritResult) and 0 <= res.score <= 1

def test_insufficient_history_fails_safe():
    short = _price([100, 101, 102])
    r = momentum_12_1_positive().evaluate(short, None)
    assert not r.passed and r.score == 0.0  # no crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_criteria.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/screen/__init__.py
```

```python
# hub/screen/criteria.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class CritResult:
    passed: bool
    score: float
    value: float

def _clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x != x:
        return 0.0
    return max(0.0, min(1.0, x))

@dataclass
class Criterion:
    name: str
    kind: str            # "technical" | "fundamental"
    hard: bool
    fn: Callable         # (price_df, fundamentals) -> CritResult
    def evaluate(self, price_df, fundamentals) -> CritResult:
        try:
            return self.fn(price_df, fundamentals)
        except Exception:
            return CritResult(False, 0.0, float("nan"))

_FAIL = lambda: CritResult(False, 0.0, float("nan"))

def near_52w_high(within_pct=0.05, window=252, hard=True) -> Criterion:
    def fn(price, fund):
        c = price["close"]
        if len(c) < 20:
            return _FAIL()
        hi = float(c.tail(window).max()); last = float(c.iloc[-1])
        dist = (hi - last) / hi if hi > 0 else 1.0
        score = _clamp01(1 - dist / within_pct) if within_pct > 0 else 0.0
        return CritResult(dist <= within_pct, score, last)
    return Criterion("near_52w_high", "technical", hard, fn)

def momentum_12_1_positive(hard=True) -> Criterion:
    def fn(price, fund):
        c = price["close"]
        if len(c) < 252:
            return _FAIL()
        r = float(c.iloc[-21] / c.iloc[-252] - 1.0)   # 12 months ago -> 1 month ago
        return CritResult(r > 0, _clamp01(r / 0.5), r)
    return Criterion("momentum_12_1", "technical", hard, fn)

def above_sma(n=200, hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        c = price["close"]
        if len(c) < n:
            return _FAIL()
        s = float(pta.sma(c, length=n).iloc[-1]); last = float(c.iloc[-1])
        ratio = last / s - 1.0 if s > 0 else -1.0
        return CritResult(last > s, _clamp01(ratio / 0.2), last)
    return Criterion(f"above_sma{n}", "technical", hard, fn)

def rsi_between(lo=50, hi=70, hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        c = price["close"]
        if len(c) < 15:
            return _FAIL()
        r = float(pta.rsi(c, length=14).iloc[-1])
        if r != r:
            return _FAIL()
        mid = (lo + hi) / 2
        score = _clamp01(1 - abs(r - mid) / ((hi - lo) / 2)) if hi > lo else 0.0
        return CritResult(lo <= r <= hi, score, r)
    return Criterion("rsi_zone", "technical", hard, fn)

def macd_bullish(hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        c = price["close"]
        if len(c) < 40:
            return _FAIL()
        m = pta.macd(c)
        if m is None or len(m) == 0:
            return _FAIL()
        hcol = [x for x in m.columns if x.startswith("MACDh")][0]
        h = float(m[hcol].iloc[-1])
        if h != h:
            return _FAIL()
        return CritResult(h > 0, 1.0 if h > 0 else 0.0, h)
    return Criterion("macd_bullish", "technical", hard, fn)

def adx_above(n=25, hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        if len(price) < 30:
            return _FAIL()
        a = pta.adx(price["high"], price["low"], price["close"])
        acol = [x for x in a.columns if x.startswith("ADX_")][0]
        v = float(a[acol].iloc[-1])
        if v != v:
            return _FAIL()
        return CritResult(v >= n, _clamp01(v / 50.0), v)
    return Criterion(f"adx_above{n}", "technical", hard, fn)
```

Add `pandas-ta` to `hub/requirements.txt` (append a line).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_criteria.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/screen/__init__.py hub/screen/criteria.py hub/requirements.txt tests/hub/test_criteria.py
git commit -m "feat(hub): screening criterion model + technical criteria (pandas-ta)"
```

---

### Task 2: Fundamental criteria + library + presets

**Files:**
- Create: `hub/screen/library.py`
- Test: `tests/hub/test_screen_library.py`

**Interfaces:**
- Fundamental factories returning `Criterion(kind="fundamental")`: `pe_below(x=40)`, `peg_below(x=2)`, `growth_above(key="earnings_growth", x=0.1)`, `margin_above(key="net_margin", x=0.1)`, `mktcap_above(x=2e9)`. Each fails closed on missing data.
- `hub.screen.library.PRESETS: dict[str, list[Criterion]]` with `growth_momentum`, `value`, `quality_momentum`; `get_preset(name) -> list[Criterion]` (ValueError on unknown).

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_screen_library.py
import pytest
from hub.screen.library import (pe_below, growth_above, margin_above, mktcap_above,
                                PRESETS, get_preset)

def test_fundamental_pass_fail_and_fail_closed():
    assert pe_below(40).evaluate(None, {"pe_ratio": 22.0}).passed
    assert not pe_below(40).evaluate(None, {"pe_ratio": 55.0}).passed
    assert not pe_below(40).evaluate(None, {"pe_ratio": None}).passed  # fail closed
    assert not pe_below(40).evaluate(None, {}).passed

def test_growth_and_margin():
    assert growth_above("earnings_growth", 0.1).evaluate(None, {"earnings_growth": 0.2}).passed
    assert not margin_above("net_margin", 0.1).evaluate(None, {"net_margin": 0.05}).passed

def test_presets_resolve():
    assert set(PRESETS) >= {"growth_momentum", "value", "quality_momentum"}
    crits = get_preset("growth_momentum")
    assert any(c.kind == "technical" for c in crits) and any(c.kind == "fundamental" for c in crits)
    with pytest.raises(ValueError):
        get_preset("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_screen_library.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/screen/library.py
from .criteria import Criterion, CritResult, _clamp01, near_52w_high, \
    momentum_12_1_positive, above_sma, rsi_between

def _fund(name, key, op, threshold, hard=True, scale=None) -> Criterion:
    def fn(price, fund):
        v = (fund or {}).get(key)
        if v is None:
            return CritResult(False, 0.0, float("nan"))   # fail closed
        try:
            v = float(v)
        except (TypeError, ValueError):
            return CritResult(False, 0.0, float("nan"))
        passed = v < threshold if op == "<" else v > threshold
        if scale:
            delta = (threshold - v) if op == "<" else (v - threshold)
            score = _clamp01(delta / abs(scale) + 0.5)
        else:
            score = 1.0 if passed else 0.0
        return CritResult(passed, score, v)
    return Criterion(name, "fundamental", hard, fn)

def pe_below(x=40, hard=True):
    return _fund("pe_below", "pe_ratio", "<", x, hard, scale=x)

def peg_below(x=2, hard=True):
    return _fund("peg_below", "peg_ratio", "<", x, hard, scale=x)

def growth_above(key="earnings_growth", x=0.1, hard=True):
    return _fund(f"{key}_above", key, ">", x, hard, scale=0.3)

def margin_above(key="net_margin", x=0.1, hard=True):
    return _fund(f"{key}_above", key, ">", x, hard, scale=0.3)

def mktcap_above(x=2e9, hard=True):
    return _fund("mktcap_above", "market_cap", ">", x, hard, scale=x)

PRESETS = {
    "growth_momentum": [near_52w_high(0.07), momentum_12_1_positive(), above_sma(200),
                        growth_above("earnings_growth", 0.10), pe_below(40)],
    "value": [pe_below(15), margin_above("net_margin", 0.10),
              growth_above("revenue_growth", 0.0), mktcap_above(2e9)],
    "quality_momentum": [above_sma(200), rsi_between(50, 70),
                         margin_above("net_margin", 0.15), growth_above("earnings_growth", 0.0)],
}

def get_preset(name: str):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return PRESETS[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_screen_library.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/screen/library.py tests/hub/test_screen_library.py
git commit -m "feat(hub): fundamental criteria + preset thesis library"
```

---

### Task 3: Staged screener (technical → fundamental → rank)

**Files:**
- Create: `hub/screen/screener.py`
- Test: `tests/hub/test_screener.py`

**Interfaces:**
- `hub.screen.screener.run_screen(universe, provider, criteria, top_k=25, lookback_days=300) -> {"candidates": [...], "skipped": [...]}`. Each candidate carries `symbol, composite, criteria{name:{passed,value,score}}, subscores{name:score}, fundamentals, explanation=None`. Names failing any hard technical filter are dropped BEFORE fundamentals are fetched.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_screener.py
import numpy as np
import pandas as pd
from hub.screen.screener import run_screen
from hub.screen.criteria import above_sma
from hub.screen.library import pe_below

def _price(trend):
    n = 300
    closes = np.linspace(100, 100 + trend, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": 1e6}, index=idx)

class StubProvider:
    def __init__(self):
        self.frames = {"UP": _price(+60), "DOWN": _price(-40)}
        self.funds = {"UP": {"pe_ratio": 20.0}, "DOWN": {"pe_ratio": 20.0}}
        self.fund_calls = []
    def get_ohlcv(self, s, lookback_days): return self.frames[s]
    def get_fundamentals(self, s):
        self.fund_calls.append(s); return self.funds[s]
    def get_news(self, s, limit=5): return []

def test_technical_filter_runs_before_fundamentals():
    p = StubProvider()
    res = run_screen(["UP", "DOWN"], p, [above_sma(200), pe_below(40)], top_k=10)
    syms = [c["symbol"] for c in res["candidates"]]
    assert "UP" in syms and "DOWN" not in syms          # DOWN fails above_sma (hard)
    assert p.fund_calls == ["UP"]                        # fundamentals fetched only for survivor

def test_fundamental_hard_filter_drops():
    p = StubProvider(); p.funds["UP"] = {"pe_ratio": 99.0}
    res = run_screen(["UP"], p, [above_sma(200), pe_below(40)], top_k=10)
    assert res["candidates"] == []                       # UP passes tech but fails pe<40

def test_ranking_and_payload():
    p = StubProvider()
    res = run_screen(["UP"], p, [above_sma(200), pe_below(40)], top_k=10)
    c = res["candidates"][0]
    assert 0 <= c["composite"] <= 1 and "above_sma200" in c["criteria"]
    assert c["fundamentals"]["pe_ratio"] == 20.0 and c["explanation"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_screener.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/screen/screener.py
import numpy as np

def run_screen(universe, provider, criteria, top_k=25, lookback_days=300) -> dict:
    tech = [c for c in criteria if c.kind == "technical"]
    fund = [c for c in criteria if c.kind == "fundamental"]
    candidates, skipped = [], []
    for sym in universe:
        try:
            price = provider.get_ohlcv(sym, lookback_days)
        except Exception as e:
            skipped.append({"symbol": sym, "reason": str(e)}); continue
        if price is None or len(price) < 60:
            skipped.append({"symbol": sym, "reason": "insufficient history"}); continue
        tres = {c.name: c.evaluate(price, None) for c in tech}
        if any(c.hard and not tres[c.name].passed for c in tech):
            continue  # dropped before any fundamental fetch
        fundamentals = provider.get_fundamentals(sym) if fund else {}
        fres = {c.name: c.evaluate(price, fundamentals) for c in fund}
        if any(c.hard and not fres[c.name].passed for c in fund):
            continue
        allres = {**tres, **fres}
        score = float(np.mean([r.score for r in allres.values()])) if allres else 0.0
        candidates.append({
            "symbol": sym, "composite": score,
            "criteria": {n: {"passed": bool(r.passed), "value": r.value, "score": r.score}
                         for n, r in allres.items()},
            "subscores": {n: r.score for n, r in allres.items()},
            "fundamentals": fundamentals, "explanation": None})
    candidates.sort(key=lambda c: c["composite"], reverse=True)
    return {"candidates": candidates[:top_k], "skipped": skipped}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_screener.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/screen/screener.py tests/hub/test_screener.py
git commit -m "feat(hub): staged screener (technical filter -> fundamentals -> rank)"
```

---

### Task 4: CLI `screen` subcommand + integration + live smoke

**Files:**
- Modify: `hub/cli.py`
- Test: `tests/hub/test_screen_cli.py`

**Interfaces:**
- `hub.cli.main(["screen", "--preset", NAME, ...])`: loads the universe + default provider, runs `run_screen(universe, provider, get_preset(NAME))`, optionally `explain_top`, writes reports, prints a summary. Flags: `--preset` (required), `--top-k`, `--universe`, `--out`, `--no-explain`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_screen_cli.py
from hub.cli import main

def test_screen_cli_runs(tmp_path, monkeypatch):
    import hub.cli as cli
    import numpy as np, pandas as pd
    def _price():
        n=300; c=np.linspace(80,160,n)
        return pd.DataFrame({"open":c,"high":c*1.01,"low":c*0.99,"close":c,"volume":1e6},
                            index=pd.date_range("2024-01-01", periods=n, freq="B"))
    class P:
        def get_ohlcv(self,s,l): return _price()
        def get_fundamentals(self,s): return {"pe_ratio":20.0,"earnings_growth":0.2,"net_margin":0.2}
        def get_news(self,s,limit=5): return []
    monkeypatch.setattr(cli, "get_default_provider", lambda d: P())
    monkeypatch.setattr(cli, "load_universe", lambda name: ["AAA","BBB"])
    rc = main(["screen", "--preset", "growth_momentum", "--no-explain", "--out", str(tmp_path)])
    assert rc == 0
    assert any(p.name.startswith("watchlist_") for p in tmp_path.iterdir())

def test_screen_cli_bad_preset(tmp_path, monkeypatch):
    import hub.cli as cli
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    monkeypatch.setattr(cli, "load_universe", lambda name: [])
    assert main(["screen", "--preset", "nope", "--out", str(tmp_path)]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_screen_cli.py -v`
Expected: FAIL (`screen` subcommand unknown)

- [ ] **Step 3: Write minimal implementation**

In `hub/cli.py`: add imports near the top — `from .screen.screener import run_screen` and `from .screen.library import get_preset`, and ensure `load_universe` is imported at module scope (it is used via `from .universe import load_universe`; if not already imported there, add it). Register the `screen` subparser and handler.

```python
# in main(), add the subparser:
    sc = sub.add_parser("screen")
    sc.add_argument("--preset", required=True)
    sc.add_argument("--no-explain", action="store_true")
    sc.add_argument("--top-k", type=int)
    sc.add_argument("--universe")
    sc.add_argument("--out")
```

```python
# in main(), add the handler branch (before `return 1`):
    if args.cmd == "screen":
        cfg = HubConfig.default()
        over = {}
        if args.top_k: over["top_k"] = args.top_k
        if args.universe: over["universe"] = args.universe
        if args.out: over["out_dir"] = args.out
        if over:
            cfg = HubConfig(**{**cfg.__dict__, **over})
        try:
            criteria = get_preset(args.preset)
        except ValueError as e:
            print(e); return 1
        provider = get_default_provider(cfg.cache_dir)
        universe = load_universe(cfg.universe)
        result = run_screen(universe, provider, criteria, top_k=cfg.top_k)
        if not args.no_explain and result["candidates"]:
            import anthropic
            result = explain_top(result, provider, anthropic.Anthropic(), cfg)
        import datetime as _dt
        date_str = _dt.datetime.now().strftime("%Y%m%d")
        paths = write_reports(result, cfg, date_str)
        print(f"preset '{args.preset}': {len(result['candidates'])} passed, "
              f"{len(result['skipped'])} skipped")
        for c in result["candidates"][:10]:
            print(f"  {c['symbol']:6s} score={c['composite']:.2f}")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
```

Ensure `load_universe`, `explain_top`, `write_reports`, `HubConfig`, `get_default_provider` are imported at module scope in cli.py (most already are from the `scan` command; add `from .universe import load_universe` if missing).

- [ ] **Step 4: Run tests + live smoke**

Run: `.venv/bin/python -m pytest tests/hub/ -q`
Expected: PASS (all hub tests)

Live smoke (real OpenBB + pandas-ta on a tiny universe, no LLM):

```bash
.venv/bin/python -m hub screen --preset growth_momentum --no-explain --universe sp500_sample --top-k 10 2>&1 | grep -vE "Warning|warn" | tail -15
```
Expected: prints `preset 'growth_momentum': N passed, M skipped` with a ranked list and report paths (N may be small — that's a working screen).

- [ ] **Step 5: Commit**

```bash
git add hub/cli.py tests/hub/test_screen_cli.py
git commit -m "feat(hub): screen CLI subcommand (preset-driven, OpenBB + pandas-ta)"
```

---

## Self-Review

**Spec coverage:** Criterion model + technical criteria (§4 → Task 1) ✓; fundamental criteria + presets (§4/§5 → Task 2) ✓; staged screener with fetch-minimization (§3 → Task 3) ✓; CLI `screen` + report reuse + live (§6 → Task 4) ✓; fail-safe criteria + fail-closed fundamentals (§7 → Tasks 1,2,3) ✓; offline tests + stub provider call-count assertion (§8 → Tasks 1,3) ✓; pandas-ta dep (§9 → Task 1) ✓. Deferred per §11: Streamlit UI, custom config criteria, edgartools/quantstats — not in this step (correct).

**Placeholder scan:** No TBD/TODO; every code step has real code.

**Type consistency:** `CritResult`/`Criterion.evaluate` (Task 1) used by all criteria + the screener. Technical factories (Task 1) imported by `library.py` presets (Task 2). `get_preset -> list[Criterion]` (Task 2) consumed by the CLI (Task 4). `run_screen -> {candidates, skipped}` (Task 3) matches the shape `explain_top`/`write_reports` already consume (same as `scan`), and each candidate's `fundamentals`/`subscores` render in the existing report. CLI handler mirrors the existing `scan` handler's imports. ✓
