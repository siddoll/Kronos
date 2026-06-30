# Momentum Screen + Honest Forward-Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** A momentum/strength screen ("showing strength now", not "will rise") + a built-in forward-test that reports the *real* historical hit-rate of its picks at 1/2/4-week horizons — so the user sees the truth, not a false prediction.

**Architecture:** New price criteria + a `momentum_catalyst` preset in `hub/screen/`; `hub/screen/forward_test.py` (point-in-time screen backtest over price criteria); a UI "Reality check" panel.

**Tech Stack:** Python 3.13, the merged hub screener + UI.

## Global Constraints

- `.venv/bin/python` / `.venv/bin/pytest`. Branch `momentum-screen`.
- HONESTY: the forward-test uses ONLY `kind=="technical"` criteria (point-in-time, no lookahead). It must NOT manufacture an edge — a random-walk universe must yield hit-rate ≈ 0.5 and edge ≈ 0 (a required test). No "will rise" labels.
- Criteria reuse the existing `CritResult`/`Criterion` model (fail-safe on error/short history).
- Tests offline (synthetic prices); the UI panel verified by launch + screenshot.

---

### Task 1: `rvol_above` + `short_momentum_positive` criteria + `momentum_catalyst` preset

**Files:**
- Modify: `hub/screen/criteria.py`, `hub/screen/library.py`
- Test: `tests/hub/test_momentum_criteria.py`

**Interfaces:**
- `criteria.rvol_above(mult=1.5, window=20, hard=False)` — relative volume `vol[-1]/avg(vol, window)`.
- `criteria.short_momentum_positive(window=20, hard=True)` — `close[-1]/close[-window-1]-1`.
- `library.PRESETS["momentum_catalyst"]` = `[rvol_above(1.5, hard=False), near_52w_high(0.12), above_sma(50), short_momentum_positive(20)]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_momentum_criteria.py
import numpy as np
import pandas as pd
from hub.screen.criteria import rvol_above, short_momentum_positive
from hub.screen.library import get_preset

def _price(closes, vols=None):
    closes = np.asarray(closes, float)
    n = len(closes)
    vols = np.full(n, 1e6) if vols is None else np.asarray(vols, float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": vols}, index=idx)

def test_rvol_spike():
    vols = [1e6]*40 + [3e6]   # last bar 3x
    r = rvol_above(1.5).evaluate(_price(np.linspace(100, 110, 41), vols), None)
    assert r.passed and r.value > 2.5 and 0 <= r.score <= 1

def test_short_momentum_sign():
    up = _price(np.linspace(50, 80, 60))
    assert short_momentum_positive(20).evaluate(up, None).passed
    down = _price(np.linspace(80, 50, 60))
    assert not short_momentum_positive(20).evaluate(down, None).passed

def test_short_history_failsafe():
    r = short_momentum_positive(20).evaluate(_price([100, 101, 102]), None)
    assert not r.passed and r.score == 0.0

def test_preset_exists():
    crits = get_preset("momentum_catalyst")
    assert any(c.name == "rvol_above" for c in crits)
    assert any(c.name == "short_momentum" for c in crits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_momentum_criteria.py -v`
Expected: FAIL with `ImportError` / unknown preset

- [ ] **Step 3: Write minimal implementation**

Append to `hub/screen/criteria.py`:

```python
def rvol_above(mult=1.5, window=20, hard=False) -> Criterion:
    def fn(price, fund):
        v = price["volume"]
        if len(v) < window + 1:
            return _FAIL()
        avg = float(v.iloc[-(window + 1):-1].mean())
        if avg <= 0:
            return _FAIL()
        ratio = float(v.iloc[-1]) / avg
        return CritResult(ratio >= mult, _clamp01((ratio - 1.0) / 2.0), ratio)
    return Criterion("rvol_above", "technical", hard, fn)

def short_momentum_positive(window=20, hard=True) -> Criterion:
    def fn(price, fund):
        c = price["close"]
        if len(c) < window + 1:
            return _FAIL()
        r = float(c.iloc[-1] / c.iloc[-(window + 1)] - 1.0)
        return CritResult(r > 0, _clamp01(r / 0.20), r)
    return Criterion("short_momentum", "technical", hard, fn)
```

In `hub/screen/library.py`, import the new criteria (extend the existing `from .criteria import ...` line to include `above_sma, rvol_above, short_momentum_positive`) and add the preset to `PRESETS`:

```python
    "momentum_catalyst": [rvol_above(1.5, hard=False), near_52w_high(0.12),
                          above_sma(50), short_momentum_positive(20)],
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_momentum_criteria.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/screen/criteria.py hub/screen/library.py tests/hub/test_momentum_criteria.py
git commit -m "feat(hub): rvol + short-momentum criteria + momentum_catalyst preset"
```

---

### Task 2: `forward_test` — honest point-in-time screen backtest

**Files:**
- Create: `hub/screen/forward_test.py`
- Test: `tests/hub/test_forward_test.py`

**Interfaces:**
- `forward_test(frames, criteria, horizons=(5,10,20), step=5, warmup=252) -> dict`: uses only `kind=="technical"` hard criteria; for each rebalance date, picks = names passing all hard tech filters point-in-time; per horizon reports `hit_rate, pick_return, market_return, edge, n`. Returns `{"horizons": {h: {...}}, "n_dates", "n_names"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_forward_test.py
import numpy as np
import pandas as pd
from hub.screen.forward_test import forward_test
from hub.screen.criteria import short_momentum_positive, above_sma

def _frame(closes):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2022-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"open": closes, "high": closes*1.01, "low": closes*0.99,
                         "close": closes, "volume": 1e6}, index=idx)

CRIT = [short_momentum_positive(20), above_sma(50)]

def test_trending_name_is_picked_and_up():
    frames = {"UP": _frame(np.linspace(50, 150, 300)), "FLAT": _frame([100.0]*300)}
    r = forward_test(frames, CRIT, horizons=(5, 10), step=10, warmup=60)
    assert r["n_dates"] > 0
    assert set(r["horizons"][5]) == {"hit_rate", "pick_return", "market_return", "edge", "n"}
    assert r["horizons"][5]["hit_rate"] > 0.6     # the persistent uptrend keeps rising

def test_random_walk_has_no_edge():
    # HONESTY GUARD: noise must yield ~coin-flip hit-rate and ~0 edge — the harness
    # must not manufacture an edge.
    rng = np.random.RandomState(0)
    frames = {}
    for i in range(20):
        steps = rng.normal(0, 0.01, 320)
        frames[f"T{i}"] = _frame(100 * np.exp(np.cumsum(steps)))
    r = forward_test(frames, CRIT, horizons=(5, 10), step=5, warmup=60)
    assert 0.40 < r["horizons"][5]["hit_rate"] < 0.60
    assert abs(r["horizons"][5]["edge"]) < 0.03

def test_empty_is_safe():
    r = forward_test({}, CRIT, horizons=(5,))
    assert r["n_dates"] == 0 and r["horizons"][5]["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_forward_test.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/screen/forward_test.py
import numpy as np

def _zero():
    return {"hit_rate": 0.0, "pick_return": 0.0, "market_return": 0.0, "edge": 0.0, "n": 0}

def forward_test(frames, criteria, horizons=(5, 10, 20), step=5, warmup=252) -> dict:
    hard_tech = [c for c in criteria if c.kind == "technical" and c.hard]
    horizons = tuple(horizons)
    if not frames or not hard_tech:
        return {"horizons": {h: _zero() for h in horizons}, "n_dates": 0,
                "n_names": len(frames)}
    n_bars = max(len(df) for df in frames.values())
    max_h = max(horizons)
    pick_rets = {h: [] for h in horizons}
    mkt_rets = {h: [] for h in horizons}
    hits = {h: [] for h in horizons}
    n_dates = 0
    for origin in range(warmup, n_bars - max_h, step):
        fwd = {h: {} for h in horizons}
        picks = []
        any_data = False
        for t, df in frames.items():
            if origin >= len(df):
                continue
            window = df.iloc[:origin]
            if len(window) < warmup:
                continue
            c0 = df["close"].iloc[origin - 1]
            got = {}
            for h in horizons:
                j = origin - 1 + h
                if j < len(df) and c0 > 0:
                    got[h] = float(df["close"].iloc[j] / c0 - 1.0)
            if not got:
                continue
            any_data = True
            for h, r in got.items():
                fwd[h][t] = r
            if all(c.evaluate(window, None).passed for c in hard_tech):
                picks.append(t)
        if not any_data:
            continue
        n_dates += 1
        for h in horizons:
            allr = list(fwd[h].values())
            if not allr:
                continue
            mkt = float(np.mean(allr))
            pr = [fwd[h][t] for t in picks if t in fwd[h]]
            if pr:
                pick_rets[h].append(float(np.mean(pr)))
                mkt_rets[h].append(mkt)
                hits[h].extend([1.0 if x > 0 else 0.0 for x in pr])
    out = {"horizons": {}, "n_dates": n_dates, "n_names": len(frames)}
    for h in horizons:
        if pick_rets[h]:
            p, m = float(np.mean(pick_rets[h])), float(np.mean(mkt_rets[h]))
            out["horizons"][h] = {"hit_rate": float(np.mean(hits[h])), "pick_return": p,
                                  "market_return": m, "edge": p - m, "n": len(hits[h])}
        else:
            out["horizons"][h] = _zero()
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_forward_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/screen/forward_test.py tests/hub/test_forward_test.py
git commit -m "feat(hub): honest point-in-time forward-test for a screen (multi-horizon)"
```

---

### Task 3: Reality-check table helper + UI panel

**Files:**
- Modify: `hub/ui/screen_runner.py` (add `forward_test_table`), `hub/ui/app.py`
- Test: `tests/hub/test_forward_table.py`

**Interfaces:**
- `screen_runner.forward_test_table(ft: dict) -> pandas.DataFrame` — rows per horizon labeled "1 week"/"2 weeks"/"4 weeks" (5/10/20 bars → those labels; other → "{h} bars"), columns `Horizon, Picks up %, Avg pick, Avg market, Edge, n`.
- UI: a `st.expander("⚖️ Reality check — how reliable is this screen?")` that runs `forward_test` for the active screen's criteria over the universe (cached) and shows the table + an honest verdict caption.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_forward_table.py
from hub.ui.screen_runner import forward_test_table

def test_forward_table_labels_and_columns():
    ft = {"horizons": {5: {"hit_rate": 0.53, "pick_return": 0.012, "market_return": 0.010,
                           "edge": 0.002, "n": 120},
                       20: {"hit_rate": 0.55, "pick_return": 0.03, "market_return": 0.028,
                            "edge": 0.002, "n": 100}},
          "n_dates": 40, "n_names": 30}
    df = forward_test_table(ft)
    assert list(df.columns) == ["Horizon", "Picks up %", "Avg pick", "Avg market", "Edge", "n"]
    horizons = set(df["Horizon"])
    assert "1 week" in horizons and "4 weeks" in horizons
    assert df.iloc[0]["Picks up %"] == "53.0%"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_forward_table.py -v`
Expected: FAIL (`forward_test_table` missing)

- [ ] **Step 3: Write minimal implementation**

Append to `hub/ui/screen_runner.py`:

```python
_HORIZON_LABEL = {5: "1 week", 10: "2 weeks", 20: "4 weeks"}

def forward_test_table(ft: dict) -> pd.DataFrame:
    rows = []
    for h, m in (ft.get("horizons") or {}).items():
        rows.append({
            "Horizon": _HORIZON_LABEL.get(h, f"{h} bars"),
            "Picks up %": f"{m['hit_rate'] * 100:.1f}%",
            "Avg pick": f"{m['pick_return'] * 100:+.2f}%",
            "Avg market": f"{m['market_return'] * 100:+.2f}%",
            "Edge": f"{m['edge'] * 100:+.2f}%",
            "n": m["n"],
        })
    return pd.DataFrame(rows, columns=["Horizon", "Picks up %", "Avg pick",
                                       "Avg market", "Edge", "n"])
```

In `hub/ui/app.py`:
1. Import: extend the screen_runner import to include `forward_test_table`; `from hub.screen.forward_test import forward_test`.
2. Add a cached forward-test runner near `_run`:

```python
@st.cache_data(show_spinner=False)
def _forward(universe_name, preset, pe_max, eps_growth_min, near_high_pct):
    universe = load_universe(universe_name)
    provider = _provider()
    frames = {}
    for s in universe:
        try:
            f = provider.get_ohlcv(s, 400)
            if f is not None and len(f) > 280:
                frames[s] = f
        except Exception:
            pass
    criteria = build_criteria(preset, {"pe_max": pe_max, "eps_growth_min": eps_growth_min,
                                       "near_high_pct": near_high_pct})
    return forward_test(frames, criteria, horizons=(5, 10, 20))
```

3. After the watchlist `st.dataframe(...)` block (inside `else:`), add the panel:

```python
    with st.expander("⚖️ Reality check — how reliable is this screen? (honest backtest)"):
        st.caption("These picks show strength NOW — this is **not** a prediction that they "
                   "will rise. Below: how this screen's price filters actually performed "
                   "historically (point-in-time, no lookahead).")
        ft = _forward(universe_name, preset, pe_max, eps_growth_min, near_high_pct)
        if ft["n_dates"] == 0 or not any(v["n"] for v in ft["horizons"].values()):
            st.info("Not enough history to backtest this screen.")
        else:
            st.dataframe(forward_test_table(ft), use_container_width=True, hide_index=True)
            e20 = ft["horizons"].get(20, {})
            hr = e20.get("hit_rate", 0) * 100
            edge = e20.get("edge", 0) * 100
            verdict = ("essentially a coin flip — no reliable edge"
                       if abs(edge) < 0.3 or 47 <= hr <= 53 else
                       f"a small historical edge of {edge:+.2f}% (treat with skepticism)")
            st.markdown(f"**At 4 weeks: picks rose {hr:.0f}% of the time → {verdict}.** "
                        "Use this as a starting point for your own research, not a buy signal.")
```

- [ ] **Step 4: Run tests + verify app imports**

Run: `.venv/bin/python -m pytest tests/hub/ -q && .venv/bin/python -c "import hub.ui.app"`
Expected: all hub tests PASS; `import hub.ui.app` does not raise.

- [ ] **Step 5: Commit**

```bash
git add hub/ui/screen_runner.py hub/ui/app.py tests/hub/test_forward_table.py
git commit -m "feat(hub): UI reality-check panel — honest forward-test of the active screen"
```

---

## Self-Review

**Spec coverage:** rvol + short-momentum criteria + momentum_catalyst preset (§4/§5 → Task 1) ✓; point-in-time multi-horizon forward-test, technical-only, no manufactured edge (§6 → Task 2) ✓; reality-check UI panel with honest verdict (§7 → Task 3) ✓; fail-safe / empty handling (§8 → Tasks 1,2) ✓; offline tests incl. the random-walk honesty guard + UI screenshot (§9 → Tasks 1-3) ✓. Deferred per §11: point-in-time catalyst feed, cost-adjusted returns — not in this step (correct).

**Placeholder scan:** No TBD/TODO; every code step has real code.

**Type consistency:** new criteria reuse `Criterion`/`CritResult`/`_clamp01`/`_FAIL` (Task 1) consumed by `forward_test` (Task 2) and the preset. `forward_test -> {horizons:{h:{hit_rate,pick_return,market_return,edge,n}}, n_dates, n_names}` (Task 2) consumed by `forward_test_table` (Task 3) and the UI. `momentum_catalyst` preset (Task 1) selectable in the existing UI preset dropdown (PRESET_NAMES auto-includes it). ✓
