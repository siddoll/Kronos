# Phase B — Orthogonal Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add two orthogonal, point-in-time, non-price features — analyst-revision momentum (`rev_mom`) and earnings-drift/PEAD (`pead`) — to the alpha panel and test, against the same honest bar, whether they add cross-sectional alpha the price signals lacked.

**Architecture:** Extend the merged `alpha/` engine. New `alpha/data.py` (cached yfinance external data) and `alpha/features/` (point-in-time builders); `build_panel` gains an optional `ext_provider`; `run_backtest` reports each new feature's IC plus a price-only vs price+orthogonal out-of-sample combiner comparison. `hub/` is untouched.

**Tech Stack:** Python 3.13 (`.venv`), pandas, numpy, scipy, lightgbm, yfinance, lxml (for earnings dates), plus the existing `alpha/`.

## Global Constraints

- `.venv/bin/python` / `.venv/bin/pytest`. Branch `phase-b-features`. Do NOT modify `hub/`.
- Backward compatibility: with `ext_provider=None` / `extra_features=()`, `build_panel` and `run_backtest` behave exactly as Phase A — all existing `tests/alpha/` stay green.
- No lookahead: a feature at date `d` uses only external events dated `≤ d`.
- Normalized external frames: `get_upgrades_downgrades(t) → DataFrame[date(datetime64), up(int), down(int)]`; `get_earnings(t) → DataFrame[date(datetime64), surprise(float)]`.
- NaN orthogonal values are median-filled per train-fold inside the combiner (no leakage); a missing value never drops a name.
- Tests use synthetic frames — no network. Only `alpha run --with-orthogonal` hits yfinance.
- Add `lxml` to `alpha/requirements.txt`.

---

### Task 1: ExternalDataProvider (cached yfinance revisions + earnings)

**Files:**
- Create: `alpha/data.py`
- Modify: `alpha/requirements.txt` (add `lxml`)
- Test: `tests/alpha/test_data.py`

**Interfaces:**
- Produces `alpha.data.ExternalDataProvider(cache_dir=".alpha_ext_cache")` with `get_upgrades_downgrades(ticker) -> pd.DataFrame[date,up,down]` and `get_earnings(ticker) -> pd.DataFrame[date,surprise]`, each cached to parquet (TTL 72h). Static normalizers `_norm_ud(raw)` and `_norm_earn(raw)` convert yfinance shapes to the normalized frames and tolerate None/empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_data.py
import pandas as pd
from alpha.data import ExternalDataProvider as EP

def test_norm_ud_classifies_up_down():
    raw = pd.DataFrame(
        {"Action": ["up", "down", "main"], "priceTargetAction": ["", "", "Raises"]},
        index=pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]))
    raw.index.name = "GradeDate"
    out = EP._norm_ud(raw)
    assert list(out.columns) == ["date", "up", "down"]
    assert out["up"].sum() == 2 and out["down"].sum() == 1  # up, Raises -> up; down -> down

def test_norm_ud_handles_empty():
    out = EP._norm_ud(None)
    assert list(out.columns) == ["date", "up", "down"] and len(out) == 0

def test_norm_earn_extracts_surprise():
    raw = pd.DataFrame({"Surprise(%)": [3.5, -1.2]},
                       index=pd.to_datetime(["2025-01-30", "2024-10-30"]))
    raw.index.name = "Earnings Date"
    out = EP._norm_earn(raw)
    assert list(out.columns) == ["date", "surprise"]
    assert set(out["surprise"]) == {3.5, -1.2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/data.py
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
```

Add `lxml` to `alpha/requirements.txt` (append a line).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/data.py alpha/requirements.txt tests/alpha/test_data.py
git commit -m "feat(alpha): external data provider (cached yfinance revisions + earnings)"
```

---

### Task 2: Point-in-time feature builders (revisions + earnings)

**Files:**
- Create: `alpha/features/__init__.py`, `alpha/features/revisions.py`, `alpha/features/earnings.py`
- Test: `tests/alpha/test_features.py`

**Interfaces:**
- `alpha.features.revisions.revision_momentum(ud_df, as_of_dates, window_days=90) -> dict[Timestamp,float]`: net (up−down) revisions with `date ∈ (d−window, d]`; NaN if none.
- `alpha.features.earnings.earnings_drift(earn_df, as_of_dates, decay_days=60) -> dict[Timestamp,float]`: most recent surprise with `date ≤ d`, decayed `exp(−days_since/decay_days)`; NaN if none.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_features.py
import numpy as np
import pandas as pd
from alpha.features.revisions import revision_momentum
from alpha.features.earnings import earnings_drift

def _ud(dates, ups, downs):
    return pd.DataFrame({"date": pd.to_datetime(dates), "up": ups, "down": downs})

def test_revision_momentum_net_and_window():
    ud = _ud(["2025-01-05", "2025-01-20", "2025-03-15"], [1, 1, 0], [0, 0, 1])
    out = revision_momentum(ud, [pd.Timestamp("2025-02-01")], window_days=90)
    assert out[pd.Timestamp("2025-02-01")] == 2.0  # two ups in window, the March down excluded

def test_revision_momentum_no_lookahead():
    ud = _ud(["2025-06-01"], [1], [0])  # AFTER the as-of date
    out = revision_momentum(ud, [pd.Timestamp("2025-02-01")], window_days=90)
    assert np.isnan(out[pd.Timestamp("2025-02-01")])  # future revision must not leak

def test_earnings_drift_uses_last_prior_and_decays():
    earn = pd.DataFrame({"date": pd.to_datetime(["2025-01-10", "2024-10-10"]),
                         "surprise": [4.0, 1.0]})
    near = earnings_drift(earn, [pd.Timestamp("2025-01-20")], decay_days=60)[pd.Timestamp("2025-01-20")]
    far = earnings_drift(earn, [pd.Timestamp("2025-03-10")], decay_days=60)[pd.Timestamp("2025-03-10")]
    assert near > far > 0  # positive surprise, decaying with days-since

def test_earnings_drift_no_prior_is_nan():
    earn = pd.DataFrame({"date": pd.to_datetime(["2025-06-01"]), "surprise": [3.0]})
    out = earnings_drift(earn, [pd.Timestamp("2025-02-01")], decay_days=60)
    assert np.isnan(out[pd.Timestamp("2025-02-01")])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/features/__init__.py
```

```python
# alpha/features/revisions.py
import numpy as np
import pandas as pd

def revision_momentum(ud_df, as_of_dates, window_days: int = 90) -> dict:
    if ud_df is None or len(ud_df) == 0:
        return {pd.Timestamp(d): np.nan for d in as_of_dates}
    dt = pd.to_datetime(ud_df["date"])
    out = {}
    for d in as_of_dates:
        d = pd.Timestamp(d)
        lo = d - pd.Timedelta(days=window_days)
        m = (dt > lo) & (dt <= d)
        out[d] = float(ud_df.loc[m, "up"].sum() - ud_df.loc[m, "down"].sum()) if m.any() else np.nan
    return out
```

```python
# alpha/features/earnings.py
import numpy as np
import pandas as pd

def earnings_drift(earn_df, as_of_dates, decay_days: int = 60) -> dict:
    if earn_df is None or len(earn_df) == 0:
        return {pd.Timestamp(d): np.nan for d in as_of_dates}
    e = earn_df.dropna(subset=["surprise"]).copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values("date")
    out = {}
    for d in as_of_dates:
        d = pd.Timestamp(d)
        prior = e[e["date"] <= d]
        if len(prior) == 0:
            out[d] = np.nan
        else:
            last = prior.iloc[-1]
            days = max(0, (d - pd.Timestamp(last["date"])).days)
            out[d] = float(last["surprise"] * np.exp(-days / decay_days))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/features/ tests/alpha/test_features.py
git commit -m "feat(alpha): point-in-time revision-momentum and earnings-drift features"
```

---

### Task 3: Panel extension (optional external features)

**Files:**
- Modify: `alpha/panel.py`
- Test: `tests/alpha/test_panel_ext.py`

**Interfaces:**
- `build_panel(provider, cfg, ext_provider=None)`: when `ext_provider` is given, adds `rev_mom` and `pead` columns (point-in-time, per ticker over the rebalance dates); when None, behaves exactly as before (no new columns).

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_panel_ext.py
import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.panel import build_panel

class TrendProvider:
    def get_ohlcv(self, symbol, lookback_days):
        n = 400
        close = np.linspace(50, 110, n)
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        return pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                             "close": close, "volume": 1e6}, index=idx)
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {}

class StubExt:
    def get_upgrades_downgrades(self, t):
        return pd.DataFrame({"date": pd.to_datetime(["2021-06-01", "2021-09-01"]),
                             "up": [1, 1], "down": [0, 0]})
    def get_earnings(self, t):
        return pd.DataFrame({"date": pd.to_datetime(["2021-05-01", "2021-08-01"]),
                             "surprise": [3.0, 2.0]})

def test_panel_without_ext_has_no_extra_cols(monkeypatch):
    import alpha.panel as pmod
    monkeypatch.setattr(pmod, "load_universe", lambda name: ["AAA", "BBB"])
    cfg = AlphaConfig(history_days=400, warmup=60, universe="x")
    panel = build_panel(TrendProvider(), cfg)
    assert "rev_mom" not in panel.columns and "pead" not in panel.columns

def test_panel_with_ext_adds_features(monkeypatch):
    import alpha.panel as pmod
    monkeypatch.setattr(pmod, "load_universe", lambda name: ["AAA", "BBB"])
    cfg = AlphaConfig(history_days=400, warmup=60, universe="x")
    panel = build_panel(TrendProvider(), cfg, ext_provider=StubExt())
    assert "rev_mom" in panel.columns and "pead" in panel.columns
    assert panel["pead"].notna().any()  # at least some dates have a prior earnings
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_panel_ext.py -v`
Expected: FAIL (`rev_mom` not added)

- [ ] **Step 3: Write minimal implementation**

Modify `alpha/panel.py`: add the `ext_provider` parameter and feature attachment. Replace the function signature and the row-building loop. The full new `build_panel`:

```python
# alpha/panel.py  (replace the build_panel function)
import numpy as np
import pandas as pd
from .universe import load_universe
from hub.signals import SIGNALS
from .sectors import load_sectors
from .features.revisions import revision_momentum
from .features.earnings import earnings_drift

def build_panel(provider, cfg, ext_provider=None) -> pd.DataFrame:
    sectors = load_sectors()
    frames = {}
    for t in load_universe(cfg.universe):
        try:
            df = provider.get_ohlcv(t, cfg.history_days)
            if df is not None and len(df) > cfg.warmup + cfg.horizon:
                frames[t] = df
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    calendar = sorted(set().union(*[set(df.index) for df in frames.values()]))
    reb_dates = [pd.Timestamp(d) for d in calendar[cfg.warmup::cfg.horizon]]

    ext = {}
    if ext_provider is not None:
        for t in frames:
            try:
                ud = ext_provider.get_upgrades_downgrades(t)
                ea = ext_provider.get_earnings(t)
            except Exception:
                ud, ea = None, None
            ext[t] = (revision_momentum(ud, reb_dates), earnings_drift(ea, reb_dates))

    rows = []
    for d in reb_dates:
        for t, df in frames.items():
            pos = df.index.searchsorted(d, side="right") - 1
            if pos < cfg.warmup or pos + cfg.horizon >= len(df):
                continue
            c0 = df["close"].iloc[pos]
            c1 = df["close"].iloc[pos + cfg.horizon]
            if c0 <= 0:
                continue
            feats = {s.name: float(s.compute(df.iloc[:pos + 1])) for s in SIGNALS}
            row = {"date": d, "ticker": t, "sector": sectors.get(t, "UNKNOWN"),
                   **feats, "fwd_ret": c1 / c0 - 1.0}
            if ext_provider is not None:
                row["rev_mom"] = ext[t][0].get(d, np.nan)
                row["pead"] = ext[t][1].get(d, np.nan)
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["fwd_ret"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/alpha/test_panel_ext.py tests/alpha/test_panel.py -v`
Expected: PASS (new + original panel tests)

- [ ] **Step 5: Commit**

```bash
git add alpha/panel.py tests/alpha/test_panel_ext.py
git commit -m "feat(alpha): optional external (orthogonal) features in panel builder"
```

---

### Task 4: Combiner NaN-fill + backtest price-only vs price+orthogonal

**Files:**
- Modify: `alpha/combine.py` (per-fold median fill), `alpha/config.py` (`extra_features`), `alpha/backtest.py` (extras IC + `_ext` combiners)
- Test: `tests/alpha/test_backtest_ext.py`

**Interfaces:**
- `alpha.combine._oos` median-fills NaN features using TRAIN-fold medians (no leakage); `composite_score` unaffected (price feats have no NaN).
- `AlphaConfig.extra_features: tuple = ()`.
- `run_backtest(panel, cfg)`: `feats_price = SIGNAL_NAMES`; if `cfg.extra_features` present in panel, also `feats_all = price + extras`, add `ic[<each extra>]`, and `ic["linear_oos_ext"]`, `ic["lgbm_oos_ext"]` computed on `feats_all`. No extras → identical to Phase A.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_backtest_ext.py
import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest
from hub.config import SIGNAL_NAMES

def _panel_with_planted_orthogonal(seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=40, freq="21D")
    tickers = [f"T{i:02d}" for i in range(30)]
    sectors = {t: ["A", "B", "C"][i % 3] for i, t in enumerate(tickers)}
    rows = []
    for d in dates:
        sec = {s: rng.normal(0, 0.05) for s in set(sectors.values())}
        mkt = rng.normal(0, 0.03)
        for t in tickers:
            feats = {n: rng.normal() for n in SIGNAL_NAMES}
            rev = rng.normal()
            fwd = mkt + sec[sectors[t]] + 0.05 * rev + rng.normal(0, 0.02)  # rev_mom predicts
            rows.append({"date": d, "ticker": t, "sector": sectors[t], **feats,
                         "rev_mom": rev, "pead": rng.normal(), "fwd_ret": fwd})
    return pd.DataFrame(rows)

def test_orthogonal_feature_adds_oos_value():
    panel = _panel_with_planted_orthogonal()
    cfg = AlphaConfig(n_folds=3, purge=1, extra_features=("rev_mom", "pead"))
    r = run_backtest(panel, cfg)
    assert r["ic"]["rev_mom"]["mean_ic"] > 0.05                       # standalone signal
    assert "lgbm_oos_ext" in r["ic"]
    assert r["ic"]["lgbm_oos_ext"]["mean_ic"] > r["ic"]["lgbm_oos"]["mean_ic"]  # extras help OOS

def test_backtest_ext_nan_features_handled():
    panel = _panel_with_planted_orthogonal()
    panel.loc[panel.index[:50], "rev_mom"] = np.nan  # inject missing
    cfg = AlphaConfig(n_folds=3, purge=1, extra_features=("rev_mom", "pead"))
    r = run_backtest(panel, cfg)  # must not crash
    assert r["ic"]["lgbm_oos_ext"]["n"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_backtest_ext.py -v`
Expected: FAIL (`extra_features` unknown / `lgbm_oos_ext` missing)

- [ ] **Step 3: Write minimal implementation**

In `alpha/combine.py`, make `_oos` median-fill per fold (replace the `_oos` function):

```python
# alpha/combine.py  (replace _oos)
def _oos(panel, target, folds, feats, fit_predict):
    pred = np.full(len(panel), np.nan)
    date = panel["date"].values.astype("datetime64[ns]")
    X = panel[feats].values.astype(float)
    y = np.asarray(target, float)
    for train_dates, test_dates in folds:
        tr = np.isin(date, np.array(train_dates, dtype="datetime64[ns]"))
        te = np.isin(date, np.array(test_dates, dtype="datetime64[ns]"))
        if tr.sum() < 10 or te.sum() == 0:
            continue
        med = np.nanmedian(X[tr], axis=0)          # train-fold medians (no leakage)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[te]), med, X[te])
        pred[te] = fit_predict(Xtr, y[tr], Xte)
    return pred
```

In `alpha/config.py`, add the field (place after `min_names`):

```python
    extra_features: tuple = ()
```

In `alpha/backtest.py`, replace the combiner/IC section so extras are scored and a full-feature combiner is added. Replace the body from `feats = list(SIGNAL_NAMES)` through the `lin`/`lgb` IC loop (the lines that build `ic` for the signals, composite, and the two OOS combiners — but NOT the later `score_map`/portfolio block) with:

```python
    feats = list(SIGNAL_NAMES)                                  # 7 price signals (name kept for the score_map below)
    extras = [f for f in cfg.extra_features if f in panel.columns]
    resid = neutralize(panel).values
    dates = panel["date"].values
    folds = purged_walk_forward(dates, n_folds=cfg.n_folds, purge=cfg.purge)

    ic = {}
    for f in feats + extras:
        ic[f] = ic_summary(rank_ic_series(dates, panel[f].values, resid))
    comp = composite_score(panel, cfg.weights)
    ic["composite"] = ic_summary(rank_ic_series(dates, comp, resid))

    def _oos_ic(pred):
        m = ~np.isnan(pred)
        return ic_summary(rank_ic_series(dates[m], pred[m], resid[m]))

    lin = linear_oos(panel, resid, folds, feats)
    lgb = lgbm_oos(panel, resid, folds, feats)
    ic["linear_oos"] = _oos_ic(lin)
    ic["lgbm_oos"] = _oos_ic(lgb)
    if extras:
        feats_all = feats + extras
        ic["linear_oos_ext"] = _oos_ic(linear_oos(panel, resid, folds, feats_all))
        lgb = lgbm_oos(panel, resid, folds, feats_all)  # use the richer model for the portfolio
        ic["lgbm_oos_ext"] = _oos_ic(lgb)
```

IMPORTANT: keep the variable named `feats` (the 7 price signals) — the later `score_map = {f: panel[f].values for f in feats}` block and the portfolio/DSR block stay exactly as in Phase A. `lgb` now refers to the richer model when extras are present, so the lgbm portfolio reflects price+orthogonal. Do not otherwise change the return statement.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/alpha/test_backtest_ext.py tests/alpha/test_backtest.py -v`
Expected: PASS (new + original backtest tests)

- [ ] **Step 5: Commit**

```bash
git add alpha/combine.py alpha/config.py alpha/backtest.py tests/alpha/test_backtest_ext.py
git commit -m "feat(alpha): NaN-safe combiner + price-only vs price+orthogonal OOS comparison"
```

---

### Task 5: CLI `--with-orthogonal` + integration test + README + live note

**Files:**
- Modify: `alpha/cli.py`, `alpha/README.md`
- Test: `tests/alpha/test_integration_ext.py`

**Interfaces:**
- `alpha.cli.main(["run", "--with-orthogonal", ...])`: sets `cfg.extra_features=("rev_mom","pead")` and passes `ext_provider=ExternalDataProvider(cfg.cache_dir + "_ext")` to `build_panel`; prints the extras' IC and the price-only vs `_ext` combiner comparison (already in the printed IC loop). Without the flag, behaves as before.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_integration_ext.py
import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest
from alpha.report import write_report
from hub.config import SIGNAL_NAMES

def _panel(seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=40, freq="21D")
    tickers = [f"T{i:02d}" for i in range(30)]
    sectors = {t: ["A", "B", "C"][i % 3] for i, t in enumerate(tickers)}
    rows = []
    for d in dates:
        sec = {s: rng.normal(0, 0.05) for s in set(sectors.values())}
        mkt = rng.normal(0, 0.03)
        for t in tickers:
            feats = {n: rng.normal() for n in SIGNAL_NAMES}
            rev = rng.normal()
            rows.append({"date": d, "ticker": t, "sector": sectors[t], **feats,
                         "rev_mom": rev, "pead": rng.normal(),
                         "fwd_ret": mkt + sec[sectors[t]] + 0.05*rev + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)

def test_ext_end_to_end_report(tmp_path):
    cfg = AlphaConfig(out_dir=str(tmp_path), n_folds=3, purge=1,
                      extra_features=("rev_mom", "pead"))
    result = run_backtest(_panel(), cfg)
    paths = write_report(result, _panel(), cfg, "20260628")
    html = open(paths["html"]).read()
    assert "rev_mom" in html and "lgbm_oos_ext" in html

def test_cli_with_orthogonal_monkeypatched(tmp_path, monkeypatch):
    import alpha.cli as cli
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    monkeypatch.setattr(cli, "build_panel",
                        lambda provider, cfg, ext_provider=None: _panel())
    rc = cli.main(["run", "--with-orthogonal", "--out", str(tmp_path)])
    assert rc == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_integration_ext.py -v`
Expected: FAIL (`--with-orthogonal` unknown)

- [ ] **Step 3: Write minimal implementation**

In `alpha/cli.py`: import `ExternalDataProvider`, add the flag, wire it. Replace the `run` subparser/handler additions:

```python
# in main(), add to the `run` subparser:
    r.add_argument("--with-orthogonal", action="store_true")
```

```python
# in the run handler, replace the panel-building lines:
    ext_provider = None
    if args.with_orthogonal:
        from .data import ExternalDataProvider
        cfg = AlphaConfig(**{**cfg.__dict__, "extra_features": ("rev_mom", "pead")})
        ext_provider = ExternalDataProvider(cfg.cache_dir + "_ext")
    provider = get_default_provider(cfg.cache_dir)
    panel = build_panel(provider, cfg, ext_provider=ext_provider)
```

Append to `alpha/README.md`:

```markdown

## Orthogonal features (Phase B)

    .venv/bin/python -m alpha run --with-orthogonal

Adds two point-in-time non-price features — analyst-revision momentum (`rev_mom`)
and earnings-drift/PEAD (`pead`, from yfinance; needs `lxml`) — and reports their
standalone Rank-IC plus a **price-only vs price+orthogonal** out-of-sample combiner
comparison (`lgbm_oos` vs `lgbm_oos_ext`).

> Short interest is snapshot-only on yfinance (no history) → not usable as a
> point-in-time feature; it needs a paid feed. The `ExternalDataProvider` interface
> is the seam to add paid revisions/short-interest/alt-data later.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/alpha/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add alpha/cli.py alpha/README.md tests/alpha/test_integration_ext.py
git commit -m "feat(alpha): alpha run --with-orthogonal + report/README + integration test"
```

---

## Self-Review

**Spec coverage:** ExternalDataProvider (§3 → Task 1) ✓; revision_momentum + earnings_drift point-in-time (§4 → Task 2) ✓; panel extension (§3 → Task 3) ✓; extras IC + price-only vs price+orthogonal OOS (§5 → Task 4) ✓; CLI `--with-orthogonal` + report + README short-interest gap note (§3/§5 → Task 5) ✓; NaN handling / backward compat (Tasks 3,4) ✓; point-in-time leakage tests (§7 → Task 2) ✓; planted-orthogonal detection test (§7 → Task 4) ✓; lxml dep (§8 → Task 1) ✓. Deferred per §10: paid short-interest/revisions, breadth expansion — not in this plan (correct).

**Placeholder scan:** No TBD/TODO; every code step has real code.

**Type consistency:** Normalized frames `[date,up,down]` / `[date,surprise]` produced in Task 1, consumed by Task 2 builders, called from Task 3 panel. `build_panel(..., ext_provider=None)` (Task 3) called by Task 5 CLI. `run_backtest` adds `rev_mom`/`pead`/`lgbm_oos_ext` ic keys (Task 4) consumed by Task 5 report/integration. `_oos` median-fill (Task 4) used by both linear_oos/lgbm_oos. `extra_features` config field (Task 4) read by backtest + set by CLI (Task 5). ✓
