# Alpha Engine (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the Discovery Hub's signals contain tradable cross-sectional alpha after market+sector neutralization, transaction costs, and multiple-testing correction — via rank-IC, purged walk-forward, a long-short portfolio, and the Deflated Sharpe Ratio.

**Architecture:** New `alpha/` package (sibling of merged `hub/`). Reuses `hub.data.provider` (cached yfinance) and `hub.signals.SIGNALS` as features. Pure-function math modules (panel → neutralize → IC → splits → combine → portfolio → stats) wired by `alpha/backtest.py` and an `alpha` CLI.

**Tech Stack:** Python 3.13 (`.venv`), pandas, numpy, scipy, matplotlib, lightgbm, scikit-learn, plus `hub/`.

## Global Constraints

- Use `.venv/bin/python` / `.venv/bin/python -m pytest`. `pandas>=2.2.3`.
- All work on branch `alpha-engine`. Do NOT modify `hub/` except importing from it.
- Reuse, don't reinvent: features come from `hub.signals.SIGNALS` (names: rvol, breakout, trend, vcp, rsi, range_exp, rel_strength); prices from `hub.data.provider.get_default_provider`.
- No lookahead: a row's features at date `d` use only price data with index ≤ `d`; the target uses prices after `d`.
- Tests must not require network. Synthetic-panel fixtures (Task 1) drive all unit tests; only the live `alpha run` CLI touches yfinance.
- Files <400 lines, one responsibility. No magic numbers in logic — all in `alpha/config.py`.
- Panel schema (used everywhere): a long `pandas.DataFrame` with a default RangeIndex and columns `["date","ticker","sector", <7 signal names>, "fwd_ret"]`; `date` is `datetime64`.
- IC arrays: helper functions take aligned numpy/Series arrays + a `dates` grouping array, never assume a particular index.
- New deps pinned in `alpha/requirements.txt`.

---

### Task 1: Scaffold — config, deps, sector map, universe, synthetic-panel fixtures

**Files:**
- Create: `alpha/__init__.py`, `alpha/config.py`, `alpha/requirements.txt`
- Create: `alpha/data/sectors.csv`, `alpha/data/alpha_sample.txt`
- Create: `tests/alpha/__init__.py`, `tests/alpha/conftest.py`
- Test: `tests/alpha/test_config.py`

**Interfaces:**
- Produces `alpha.config.AlphaConfig` (frozen dataclass): `universe="alpha_sample"`, `history_days=1300`, `horizon=21`, `warmup=60`, `n_quantiles=5`, `min_names=20`, `cost_bps=10.0`, `n_folds=4`, `purge=2`, `weights` (dict, imported from `hub.config`), `out_dir="out_alpha"`, `cache_dir=".alpha_cache"`; classmethod `default()`.
- Produces fixtures `planted_panel()` and `noise_panel()` in `tests/alpha/conftest.py` (see code) returning panels with the schema above.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_config.py
from alpha.config import AlphaConfig
from hub.config import SIGNAL_NAMES

def test_config_defaults():
    c = AlphaConfig.default()
    assert c.horizon == 21 and c.n_quantiles == 5
    assert set(c.weights) == set(SIGNAL_NAMES)

def test_fixtures_have_panel_schema(planted_panel):
    cols = set(planted_panel.columns)
    assert {"date","ticker","sector","fwd_ret"} <= cols
    assert set(SIGNAL_NAMES) <= cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alpha'`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/__init__.py
"""Alpha Engine — cross-sectional, factor-neutral backtest of hub signals."""
```

```python
# alpha/config.py
from dataclasses import dataclass, field
from hub.config import SIGNAL_NAMES, _DEFAULT_WEIGHTS

@dataclass(frozen=True)
class AlphaConfig:
    universe: str = "alpha_sample"
    history_days: int = 1300
    horizon: int = 21
    warmup: int = 60
    n_quantiles: int = 5
    min_names: int = 20
    cost_bps: float = 10.0
    n_folds: int = 4
    purge: int = 2
    weights: dict = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    out_dir: str = "out_alpha"
    cache_dir: str = ".alpha_cache"

    @classmethod
    def default(cls) -> "AlphaConfig":
        return cls()
```

```python
# alpha/requirements.txt
lightgbm
scikit-learn
# (pandas, numpy, scipy, matplotlib, yfinance come from hub/requirements.txt)
```

`alpha/data/alpha_sample.txt` — ~60 liquid large/mid-cap tickers, one per line (bound the live fetch). Start from `hub/data/sp500_sample.txt`'s 30 and add 30 more:

```text
# Alpha Engine sample universe (liquid US large/mid-cap)
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
TXN
AMGN
HON
LOW
SBUX
CAT
GS
BLK
GE
BA
MMM
PFE
T
VZ
CVX
NKE
MCD
IBM
GILD
ISRG
NOW
INTU
AMAT
MU
LRCX
ABBV
DHR
TMO
LIN
PYPL
```

`alpha/data/sectors.csv` — `ticker,sector` for every ticker above (use GICS-ish buckets). Header `ticker,sector` then one row per ticker, e.g.:

```text
ticker,sector
AAPL,Technology
MSFT,Technology
NVDA,Technology
AMZN,ConsumerDiscretionary
GOOGL,CommunicationServices
META,CommunicationServices
TSLA,ConsumerDiscretionary
AVGO,Technology
JPM,Financials
V,Financials
UNH,HealthCare
XOM,Energy
JNJ,HealthCare
WMT,ConsumerStaples
MA,Financials
PG,ConsumerStaples
HD,ConsumerDiscretionary
COST,ConsumerStaples
ORCL,Technology
NFLX,CommunicationServices
AMD,Technology
CRM,Technology
KO,ConsumerStaples
PEP,ConsumerStaples
ADBE,Technology
BAC,Financials
DIS,CommunicationServices
INTC,Technology
CSCO,Technology
QCOM,Technology
TXN,Technology
AMGN,HealthCare
HON,Industrials
LOW,ConsumerDiscretionary
SBUX,ConsumerDiscretionary
CAT,Industrials
GS,Financials
BLK,Financials
GE,Industrials
BA,Industrials
MMM,Industrials
PFE,HealthCare
T,CommunicationServices
VZ,CommunicationServices
CVX,Energy
NKE,ConsumerDiscretionary
MCD,ConsumerDiscretionary
IBM,Technology
GILD,HealthCare
ISRG,HealthCare
NOW,Technology
INTU,Technology
AMAT,Technology
MU,Technology
LRCX,Technology
ABBV,HealthCare
DHR,HealthCare
TMO,HealthCare
LIN,Materials
PYPL,Financials
```

```python
# tests/alpha/__init__.py
```

```python
# tests/alpha/conftest.py
import numpy as np
import pandas as pd
import pytest
from hub.config import SIGNAL_NAMES

def _make_panel(seed, planted):
    """Synthetic long panel. If planted, feature 'rvol' is correlated with a
    market+sector-neutral component of fwd_ret; otherwise all features are noise."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=40, freq="21D")
    tickers = [f"T{i:02d}" for i in range(30)]
    sectors = {t: ["A", "B", "C"][i % 3] for i, t in enumerate(tickers)}
    rows = []
    for d in dates:
        sec_shock = {s: rng.normal(0, 0.05) for s in set(sectors.values())}
        mkt = rng.normal(0, 0.03)
        for t in tickers:
            feats = {name: rng.normal() for name in SIGNAL_NAMES}
            alpha = 0.04 * feats["rvol"] if planted else 0.0
            fwd = mkt + sec_shock[sectors[t]] + alpha + rng.normal(0, 0.02)
            rows.append({"date": d, "ticker": t, "sector": sectors[t],
                         **feats, "fwd_ret": fwd})
    return pd.DataFrame(rows)

@pytest.fixture
def planted_panel():
    return _make_panel(0, planted=True)

@pytest.fixture
def noise_panel():
    return _make_panel(1, planted=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/__init__.py alpha/config.py alpha/requirements.txt alpha/data/ tests/alpha/__init__.py tests/alpha/conftest.py tests/alpha/test_config.py
git commit -m "feat(alpha): scaffold, config, universe+sector data, synthetic-panel fixtures"
```

---

### Task 2: Sector map loader + panel builder

**Files:**
- Create: `alpha/sectors.py`, `alpha/panel.py`
- Test: `tests/alpha/test_panel.py`

**Interfaces:**
- Produces `alpha.sectors.load_sectors() -> dict[str,str]` (reads `alpha/data/sectors.csv`; unknown ticker → caller defaults to `"UNKNOWN"`).
- Produces `alpha.panel.build_panel(provider, cfg) -> pd.DataFrame` with the panel schema. Point-in-time: for each rebalance date (the universe's union calendar sampled every `cfg.horizon` bars starting at `cfg.warmup`), for each ticker with a bar at-or-before the date and ≥ `cfg.horizon` bars ahead, compute the 7 signals on the slice up to that bar and `fwd_ret = close[pos+horizon]/close[pos]-1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_panel.py
import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.panel import build_panel
from alpha.sectors import load_sectors
from hub.config import SIGNAL_NAMES

def test_load_sectors_has_known():
    s = load_sectors()
    assert s.get("AAPL") == "Technology"

class TrendProvider:
    """Deterministic rising series per ticker, enough bars for the panel."""
    def get_ohlcv(self, symbol, lookback_days):
        n = 400
        base = 50 + (hash(symbol) % 10)
        close = np.linspace(base, base + 60, n)
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        return pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                             "close": close, "volume": 1e6}, index=idx)
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {}

def test_build_panel_schema_and_no_lookahead(monkeypatch):
    import alpha.panel as pmod
    monkeypatch.setattr(pmod, "load_universe", lambda name: ["AAA", "BBB", "CCC"])
    cfg = AlphaConfig(history_days=400, horizon=21, warmup=60, universe="x")
    panel = build_panel(TrendProvider(), cfg)
    assert {"date","ticker","sector","fwd_ret"} <= set(panel.columns)
    assert set(SIGNAL_NAMES) <= set(panel.columns)
    assert len(panel) > 0 and panel["fwd_ret"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/sectors.py
import csv, os

_PATH = os.path.join(os.path.dirname(__file__), "data", "sectors.csv")

def load_sectors() -> dict:
    out = {}
    with open(_PATH) as f:
        for row in csv.DictReader(f):
            out[row["ticker"].strip()] = row["sector"].strip()
    return out
```

```python
# alpha/panel.py
import numpy as np
import pandas as pd
from hub.universe import load_universe
from hub.signals import SIGNALS
from .sectors import load_sectors

def build_panel(provider, cfg) -> pd.DataFrame:
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
    reb_dates = calendar[cfg.warmup::cfg.horizon]
    rows = []
    for d in reb_dates:
        for t, df in frames.items():
            pos = df.index.searchsorted(d, side="right") - 1
            if pos < cfg.warmup or pos + cfg.horizon >= len(df):
                continue
            window = df.iloc[:pos + 1]
            c0 = df["close"].iloc[pos]
            c1 = df["close"].iloc[pos + cfg.horizon]
            if c0 <= 0:
                continue
            feats = {s.name: float(s.compute(window)) for s in SIGNALS}
            rows.append({"date": pd.Timestamp(d), "ticker": t,
                         "sector": sectors.get(t, "UNKNOWN"),
                         **feats, "fwd_ret": c1 / c0 - 1.0})
    return pd.DataFrame(rows).dropna(subset=["fwd_ret"]).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_panel.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/sectors.py alpha/panel.py tests/alpha/test_panel.py
git commit -m "feat(alpha): sector loader + point-in-time cross-sectional panel builder"
```

---

### Task 3: Market + sector neutralization

**Files:**
- Create: `alpha/neutralize.py`
- Test: `tests/alpha/test_neutralize.py`

**Interfaces:**
- Produces `alpha.neutralize.neutralize(panel, target="fwd_ret") -> pd.Series` aligned to `panel.index`: the target with per-(date,sector) mean removed (which also zeroes each date's overall mean).

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_neutralize.py
import numpy as np
from alpha.neutralize import neutralize

def test_neutralized_means_are_zero(planted_panel):
    resid = neutralize(planted_panel)
    df = planted_panel.assign(_r=resid.values)
    # per-date mean ~0
    assert df.groupby("date")["_r"].mean().abs().max() < 1e-9
    # per-(date,sector) mean ~0
    assert df.groupby(["date","sector"])["_r"].mean().abs().max() < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_neutralize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/neutralize.py
import pandas as pd

def neutralize(panel: pd.DataFrame, target: str = "fwd_ret") -> pd.Series:
    g = panel.groupby(["date", "sector"])[target]
    resid = panel[target] - g.transform("mean")
    return resid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_neutralize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/neutralize.py tests/alpha/test_neutralize.py
git commit -m "feat(alpha): market+sector cross-sectional neutralization"
```

---

### Task 4: Rank-IC series + summary stats

**Files:**
- Create: `alpha/ic.py`
- Test: `tests/alpha/test_ic.py`

**Interfaces:**
- Produces `alpha.ic.rank_ic_series(dates, x, y) -> pd.Series` (indexed by date): per-date Spearman rank correlation of `x` vs `y` (arrays aligned to `dates`); dates with < 3 valid pairs or NaN IC are skipped.
- Produces `alpha.ic.ic_summary(ic_series) -> dict` with `mean_ic`, `ic_ir` (mean/std, ddof=1), `t_stat` (ic_ir·√n), `n`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_ic.py
import numpy as np
import pandas as pd
from alpha.ic import rank_ic_series, ic_summary

def test_monotonic_relationship_ic_near_one():
    dates = np.repeat(pd.date_range("2022-01-01", periods=5), 10)
    x = np.tile(np.arange(10), 5).astype(float)
    y = x.copy()  # perfectly monotonic each date
    ic = rank_ic_series(dates, x, y)
    assert ic.mean() > 0.99
    s = ic_summary(ic)
    assert s["n"] == 5 and s["t_stat"] > 5

def test_planted_panel_has_positive_ic(planted_panel):
    from alpha.neutralize import neutralize
    resid = neutralize(planted_panel).values
    ic = rank_ic_series(planted_panel["date"].values, planted_panel["rvol"].values, resid)
    assert ic_summary(ic)["mean_ic"] > 0.05  # planted signal detectable

def test_noise_panel_ic_insignificant(noise_panel):
    from alpha.neutralize import neutralize
    resid = neutralize(noise_panel).values
    ic = rank_ic_series(noise_panel["date"].values, noise_panel["rvol"].values, resid)
    assert abs(ic_summary(ic)["t_stat"]) < 2.5  # not significant
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_ic.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/ic.py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def rank_ic_series(dates, x, y) -> pd.Series:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "x": np.asarray(x, float),
                       "y": np.asarray(y, float)}).dropna()
    out = {}
    for d, g in df.groupby("date"):
        if len(g) < 3:
            continue
        ic = spearmanr(g["x"], g["y"]).correlation
        if ic == ic:  # not NaN
            out[d] = ic
    return pd.Series(out, dtype=float)

def ic_summary(ic_series: pd.Series) -> dict:
    n = int(len(ic_series))
    if n == 0:
        return {"mean_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "n": 0}
    mean = float(ic_series.mean())
    std = float(ic_series.std(ddof=1)) if n > 1 else 0.0
    ir = mean / std if std > 0 else 0.0
    return {"mean_ic": mean, "ic_ir": ir, "t_stat": ir * np.sqrt(n), "n": n}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_ic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/ic.py tests/alpha/test_ic.py
git commit -m "feat(alpha): cross-sectional rank-IC series and summary stats"
```

---

### Task 5: Purged + embargoed walk-forward splits

**Files:**
- Create: `alpha/splits.py`
- Test: `tests/alpha/test_splits.py`

**Interfaces:**
- Produces `alpha.splits.purged_walk_forward(dates, n_folds, purge) -> list[(train_dates, test_dates)]`. `dates` is reduced to sorted unique. Test blocks are contiguous chronological chunks; train is the expanding window strictly before each test block, with the last `purge` rebalance dates dropped (purge+embargo combined) to prevent target-window overlap leakage.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_splits.py
import pandas as pd
from alpha.splits import purged_walk_forward

def test_folds_are_oos_and_purged():
    dates = pd.date_range("2022-01-01", periods=50, freq="21D")
    folds = purged_walk_forward(dates, n_folds=4, purge=2)
    assert len(folds) == 4
    for train, test in folds:
        assert max(train) < min(test)                       # strictly out-of-sample
        # purge gap: at least `purge` rebalance dates between last train and first test
        all_d = sorted(pd.to_datetime(dates).unique())
        gap = all_d.index(min(test)) - all_d.index(max(train))
        assert gap >= 2

def test_expanding_train_grows():
    dates = pd.date_range("2022-01-01", periods=50, freq="21D")
    folds = purged_walk_forward(dates, n_folds=4, purge=1)
    assert len(folds[1][0]) > len(folds[0][0])  # train expands
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/splits.py
import numpy as np
import pandas as pd

def purged_walk_forward(dates, n_folds: int = 4, purge: int = 2):
    udates = sorted(pd.to_datetime(pd.Series(dates)).unique())
    n = len(udates)
    fold = n // (n_folds + 1)
    if fold == 0:
        return []
    out = []
    for i in range(1, n_folds + 1):
        test_start = i * fold
        test_end = (i + 1) * fold if i < n_folds else n
        test = udates[test_start:test_end]
        train_end = max(0, test_start - purge)
        train = udates[:train_end]
        if train and test:
            out.append(([pd.Timestamp(d) for d in train],
                        [pd.Timestamp(d) for d in test]))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/splits.py tests/alpha/test_splits.py
git commit -m "feat(alpha): purged + embargoed walk-forward fold generator"
```

---

### Task 6: Composite score + linear/LightGBM out-of-sample combiners

**Files:**
- Create: `alpha/combine.py`
- Test: `tests/alpha/test_combine.py`

**Interfaces:**
- Consumes panel, `cfg.weights`, folds from Task 5, feature name list.
- Produces `composite_score(panel, weights) -> np.ndarray` (cross-sectionally z-scored features per date, weighted sum; aligned to panel rows).
- Produces `linear_oos(panel, target, folds, feats) -> np.ndarray` and `lgbm_oos(panel, target, folds, feats) -> np.ndarray`: out-of-sample predictions (NaN where a row is in no test fold), each fold fit on its train dates and predicting its test dates. `target` is an array aligned to panel rows (the neutralized target).

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_combine.py
import numpy as np
from alpha.combine import composite_score, linear_oos, lgbm_oos
from alpha.neutralize import neutralize
from alpha.splits import purged_walk_forward
from alpha.ic import rank_ic_series, ic_summary
from hub.config import SIGNAL_NAMES

def test_composite_is_per_row(planted_panel):
    s = composite_score(planted_panel, {n: 1.0 for n in SIGNAL_NAMES})
    assert len(s) == len(planted_panel)

def test_linear_oos_recovers_planted_signal(planted_panel):
    resid = neutralize(planted_panel).values
    folds = purged_walk_forward(planted_panel["date"].values, n_folds=3, purge=1)
    pred = linear_oos(planted_panel, resid, folds, list(SIGNAL_NAMES))
    mask = ~np.isnan(pred)
    ic = rank_ic_series(planted_panel["date"].values[mask], pred[mask], resid[mask])
    assert ic_summary(ic)["mean_ic"] > 0.05  # learns the rvol->target relation OOS

def test_lgbm_oos_runs(noise_panel):
    resid = neutralize(noise_panel).values
    folds = purged_walk_forward(noise_panel["date"].values, n_folds=3, purge=1)
    pred = lgbm_oos(noise_panel, resid, folds, list(SIGNAL_NAMES))
    assert pred.shape[0] == len(noise_panel)
    assert np.isnan(pred).any()  # rows before first test fold are unpredicted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_combine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/combine.py
import numpy as np
import pandas as pd

def composite_score(panel: pd.DataFrame, weights: dict) -> np.ndarray:
    feats = list(weights)
    def _z(g):
        return (g - g.mean()) / (g.std(ddof=0) + 1e-9)
    z = panel.groupby("date")[feats].transform(_z)
    score = sum(z[f].values * weights[f] for f in feats)
    return np.asarray(score, dtype=float)

def _oos(panel, target, folds, feats, fit_predict):
    pred = np.full(len(panel), np.nan)
    date = panel["date"].values
    X = panel[feats].values
    y = np.asarray(target, float)
    for train_dates, test_dates in folds:
        tr = np.isin(date, np.array(train_dates, dtype="datetime64[ns]"))
        te = np.isin(date, np.array(test_dates, dtype="datetime64[ns]"))
        if tr.sum() < 10 or te.sum() == 0:
            continue
        pred[te] = fit_predict(X[tr], y[tr], X[te])
    return pred

def linear_oos(panel, target, folds, feats) -> np.ndarray:
    def fp(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
        return B @ coef
    return _oos(panel, target, folds, feats, fp)

def lgbm_oos(panel, target, folds, feats) -> np.ndarray:
    import lightgbm as lgb
    def fp(Xtr, ytr, Xte):
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15,
                              learning_rate=0.05, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    return _oos(panel, target, folds, feats, fp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_combine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/combine.py tests/alpha/test_combine.py
git commit -m "feat(alpha): composite score + linear/LightGBM out-of-sample combiners"
```

---

### Task 7: Long-short portfolio with turnover costs

**Files:**
- Create: `alpha/portfolio.py`
- Test: `tests/alpha/test_portfolio.py`

**Interfaces:**
- Produces `alpha.portfolio.long_short(panel, score, n_quantiles, cost_bps, raw_target="fwd_ret") -> dict` with `net` (pd.Series of per-rebalance net returns by date), `gross` (Series), `turnover` (Series), `sharpe` (annualized, periods/yr = 252/horizon inferred from median date gap → use fixed 252/21 fallback), `sharpe_per_period`, `equity` (cumprod of 1+net). Per date: long top quantile, short bottom quantile (equal weight); net = gross − cost_bps/1e4 × turnover.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_portfolio.py
import numpy as np
from alpha.portfolio import long_short

def test_perfect_score_is_profitable_and_costs_reduce(planted_panel):
    # score = the realized fwd_ret (oracle) -> long winners, short losers
    score = planted_panel["fwd_ret"].values
    free = long_short(planted_panel, score, n_quantiles=5, cost_bps=0.0)
    paid = long_short(planted_panel, score, n_quantiles=5, cost_bps=50.0)
    assert free["gross"].mean() > 0
    assert paid["net"].mean() < free["net"].mean()  # costs drag

def test_random_score_centered_near_zero(noise_panel):
    rng = np.random.RandomState(3)
    score = rng.normal(size=len(noise_panel))
    r = long_short(noise_panel, score, n_quantiles=5, cost_bps=0.0)
    assert abs(r["gross"].mean()) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/portfolio.py
import numpy as np
import pandas as pd

def long_short(panel, score, n_quantiles=5, cost_bps=10.0,
               raw_target="fwd_ret", periods_per_year=12) -> dict:
    df = panel[["date", "ticker", raw_target]].copy()
    df["score"] = np.asarray(score, float)
    df = df.dropna(subset=["score"])
    gross, turn, dates = [], [], []
    prev_long, prev_short = set(), set()
    for d, g in df.groupby("date"):
        if len(g) < n_quantiles:
            continue
        g = g.sort_values("score")
        k = max(1, len(g) // n_quantiles)
        short = g.iloc[:k]
        long_ = g.iloc[-k:]
        ret = long_[raw_target].mean() - short[raw_target].mean()
        ls, ss = set(long_["ticker"]), set(short["ticker"])
        denom = len(ls) + len(ss)
        changed = len(ls ^ prev_long) + len(ss ^ prev_short)
        t = changed / denom if denom else 0.0
        prev_long, prev_short = ls, ss
        gross.append(ret); turn.append(t); dates.append(d)
    gross = pd.Series(gross, index=pd.to_datetime(dates))
    turnover = pd.Series(turn, index=gross.index)
    net = gross - cost_bps / 1e4 * turnover
    sd = net.std(ddof=1)
    spp = net.mean() / sd if sd > 0 else 0.0
    return {"gross": gross, "net": net, "turnover": turnover,
            "sharpe_per_period": spp, "sharpe": spp * np.sqrt(periods_per_year),
            "equity": (1 + net).cumprod()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_portfolio.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/portfolio.py tests/alpha/test_portfolio.py
git commit -m "feat(alpha): long-short quantile portfolio with turnover costs"
```

---

### Task 8: Deflated Sharpe Ratio + probabilistic Sharpe

**Files:**
- Create: `alpha/stats.py`
- Test: `tests/alpha/test_stats.py`

**Interfaces:**
- Produces `alpha.stats.prob_sharpe_ratio(sr, T, skew=0.0, kurt=3.0, sr_benchmark=0.0) -> float` (Bailey-LdP PSR; `sr` is per-period).
- Produces `expected_max_sharpe(sr_variance, n_trials) -> float`.
- Produces `deflated_sharpe_ratio(sr, T, skew, kurt, sr_variance, n_trials) -> float` = PSR with benchmark = expected max Sharpe of `n_trials`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_stats.py
import numpy as np
from alpha.stats import prob_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio

def test_psr_increases_with_sharpe():
    assert prob_sharpe_ratio(0.3, 200) > prob_sharpe_ratio(0.05, 200)

def test_psr_normal_reduces_to_phi():
    # skew=0, kurt=3, benchmark=0 -> Phi(sr*sqrt(T-1))
    from scipy.stats import norm
    val = prob_sharpe_ratio(0.2, 101, skew=0.0, kurt=3.0, sr_benchmark=0.0)
    assert abs(val - norm.cdf(0.2 * np.sqrt(100))) < 1e-9

def test_more_trials_lowers_dsr():
    a = deflated_sharpe_ratio(0.3, 200, 0.0, 3.0, sr_variance=0.01, n_trials=5)
    b = deflated_sharpe_ratio(0.3, 200, 0.0, 3.0, sr_variance=0.01, n_trials=500)
    assert b < a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/stats.py
import numpy as np
from scipy.stats import norm

def prob_sharpe_ratio(sr, T, skew=0.0, kurt=3.0, sr_benchmark=0.0) -> float:
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(max(1, T - 1)) / denom))

def expected_max_sharpe(sr_variance, n_trials) -> float:
    n = max(2, int(n_trials))
    g = 0.5772156649015329  # Euler-Mascheroni
    a = norm.ppf(1 - 1.0 / n)
    b = norm.ppf(1 - 1.0 / (n * np.e))
    return float(np.sqrt(max(0.0, sr_variance)) * ((1 - g) * a + g * b))

def deflated_sharpe_ratio(sr, T, skew, kurt, sr_variance, n_trials) -> float:
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    return prob_sharpe_ratio(sr, T, skew, kurt, sr0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_stats.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/stats.py tests/alpha/test_stats.py
git commit -m "feat(alpha): probabilistic + deflated Sharpe ratio"
```

---

### Task 9: Backtest orchestration

**Files:**
- Create: `alpha/backtest.py`
- Test: `tests/alpha/test_backtest.py`

**Interfaces:**
- Produces `alpha.backtest.run_backtest(panel, cfg) -> dict` (takes a prebuilt panel so it's testable offline) with:
  - `ic`: dict keyed by each signal name + `"composite"` + `"linear_oos"` + `"lgbm_oos"`, each an `ic_summary` dict.
  - `portfolio`: dict for `"composite"` and `"lgbm_oos"` with `sharpe`, `sharpe_per_period`, `mean_net`, `dsr` (deflated Sharpe; `n_trials` = number of models evaluated, `sr_variance` = variance of the models' per-period Sharpes).
  - `n_dates`, `n_rows`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_backtest.py
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest

def test_backtest_detects_planted_signal(planted_panel):
    cfg = AlphaConfig(n_folds=3, purge=1)
    r = run_backtest(planted_panel, cfg)
    assert r["ic"]["rvol"]["mean_ic"] > 0.05          # raw signal IC
    assert r["ic"]["lgbm_oos"]["n"] > 0               # OOS preds produced
    assert "dsr" in r["portfolio"]["composite"]
    assert r["n_rows"] == len(planted_panel)

def test_backtest_noise_not_significant(noise_panel):
    cfg = AlphaConfig(n_folds=3, purge=1)
    r = run_backtest(noise_panel, cfg)
    assert abs(r["ic"]["rvol"]["t_stat"]) < 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/backtest.py
import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from hub.config import SIGNAL_NAMES
from .neutralize import neutralize
from .ic import rank_ic_series, ic_summary
from .splits import purged_walk_forward
from .combine import composite_score, linear_oos, lgbm_oos
from .portfolio import long_short
from .stats import deflated_sharpe_ratio

def run_backtest(panel, cfg) -> dict:
    feats = list(SIGNAL_NAMES)
    resid = neutralize(panel).values
    dates = panel["date"].values
    folds = purged_walk_forward(dates, n_folds=cfg.n_folds, purge=cfg.purge)

    ic = {}
    for f in feats:
        ic[f] = ic_summary(rank_ic_series(dates, panel[f].values, resid))
    comp = composite_score(panel, cfg.weights)
    ic["composite"] = ic_summary(rank_ic_series(dates, comp, resid))
    lin = linear_oos(panel, resid, folds, feats)
    lgb = lgbm_oos(panel, resid, folds, feats)
    for name, pred in [("linear_oos", lin), ("lgbm_oos", lgb)]:
        m = ~np.isnan(pred)
        ic[name] = ic_summary(rank_ic_series(dates[m], pred[m], resid[m]))

    # portfolios for composite and the lgbm OOS predictions
    ppy = max(1, round(252 / cfg.horizon))
    scores = {"composite": comp, "lgbm_oos": lgb}
    spps = {}
    for name, sc in scores.items():
        spps[name] = long_short(panel, sc, cfg.n_quantiles, cfg.cost_bps,
                                periods_per_year=ppy)["sharpe_per_period"]
    sr_var = float(np.var(list(spps.values()))) or 1e-6
    n_trials = len(ic)  # configs evaluated
    portfolio = {}
    for name, sc in scores.items():
        r = long_short(panel, sc, cfg.n_quantiles, cfg.cost_bps, periods_per_year=ppy)
        net = r["net"].dropna()
        dsr = deflated_sharpe_ratio(
            r["sharpe_per_period"], len(net),
            float(_skew(net)) if len(net) > 2 else 0.0,
            float(_kurtosis(net, fisher=False)) if len(net) > 3 else 3.0,
            sr_var, n_trials)
        portfolio[name] = {"sharpe": r["sharpe"],
                           "sharpe_per_period": r["sharpe_per_period"],
                           "mean_net": float(net.mean()), "dsr": dsr}
    return {"ic": ic, "portfolio": portfolio,
            "n_dates": len(set(dates)), "n_rows": len(panel)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_backtest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/backtest.py tests/alpha/test_backtest.py
git commit -m "feat(alpha): backtest orchestration (IC table + portfolio + DSR)"
```

---

### Task 10: Report (IC table + equity PNG + metrics)

**Files:**
- Create: `alpha/report.py`
- Test: `tests/alpha/test_report.py`

**Interfaces:**
- Produces `alpha.report.write_report(result, panel, cfg, date_str) -> dict[str,str]` writing `<out>/alpha_<date>.json`, `<out>/alpha_<date>.csv` (IC table), `<out>/alpha_<date>.html`, and `<out>/equity_<date>.png` (composite long-short equity curve). Returns the paths. HTML is escaped.

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_report.py
import os
from alpha.config import AlphaConfig
from alpha.report import write_report

def test_report_files_written(tmp_path, planted_panel):
    cfg = AlphaConfig(out_dir=str(tmp_path), n_folds=3, purge=1)
    result = {
        "ic": {"rvol": {"mean_ic": 0.08, "ic_ir": 0.5, "t_stat": 3.1, "n": 30},
               "composite": {"mean_ic": 0.06, "ic_ir": 0.4, "t_stat": 2.2, "n": 30}},
        "portfolio": {"composite": {"sharpe": 0.9, "sharpe_per_period": 0.26,
                                    "mean_net": 0.004, "dsr": 0.7}},
        "n_dates": 30, "n_rows": len(planted_panel)}
    paths = write_report(result, planted_panel, cfg, "20260628")
    for key in ("json", "csv", "html", "equity_png"):
        assert os.path.exists(paths[key])
    assert "rvol" in open(paths["html"]).read()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/report.py
import os, json, csv
import html as _html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .combine import composite_score
from .portfolio import long_short

def write_report(result, panel, cfg, date_str) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = os.path.join(cfg.out_dir, f"alpha_{date_str}")
    paths = {"json": base + ".json", "csv": base + ".csv", "html": base + ".html",
             "equity_png": os.path.join(cfg.out_dir, f"equity_{date_str}.png")}

    with open(paths["json"], "w") as f:
        json.dump(result, f, indent=2, default=str)

    ic = result["ic"]
    with open(paths["csv"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_ic", "ic_ir", "t_stat", "n"])
        for name, s in ic.items():
            w.writerow([name, round(s["mean_ic"], 4), round(s["ic_ir"], 3),
                        round(s["t_stat"], 2), s["n"]])

    # equity curve for composite
    try:
        eq = long_short(panel, composite_score(panel, cfg.weights),
                        cfg.n_quantiles, cfg.cost_bps)["equity"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(eq.index, eq.values, color="#1f4eb0")
        ax.set_title("Composite long-short equity (after costs)")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(paths["equity_png"], dpi=110); plt.close(fig)
    except Exception:
        # still emit an (empty) file so the path exists
        plt.figure(); plt.savefig(paths["equity_png"]); plt.close()

    rows = "".join(
        f"<tr><td>{_html.escape(n)}</td><td>{s['mean_ic']:.4f}</td>"
        f"<td>{s['ic_ir']:.3f}</td><td>{s['t_stat']:.2f}</td><td>{s['n']}</td>"
        f"<td>{'significant' if abs(s['t_stat'])>=2 else '— noise'}</td></tr>"
        for n, s in ic.items())
    pf = result.get("portfolio", {})
    pf_rows = "".join(
        f"<tr><td>{_html.escape(k)}</td><td>{v['sharpe']:.2f}</td>"
        f"<td>{v['mean_net']:.4f}</td><td>{v['dsr']:.2f}</td>"
        f"<td>{'PASS' if v['dsr']>=0.95 else 'fail'}</td></tr>"
        for k, v in pf.items())
    html = (f"<html><head><meta charset='utf-8'><title>Alpha Engine {date_str}</title>"
            "<style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;margin-bottom:20px}"
            "td,th{border:1px solid #ddd;padding:6px;font-size:13px}"
            "th{background:#f4f4f4}</style></head><body>"
            f"<h2>Alpha Engine — {date_str}</h2>"
            f"<p>{result['n_rows']} name-dates over {result['n_dates']} rebalances. "
            "Market+sector-neutral target, out-of-sample, after costs. "
            "Significance: |t|≥2 for IC; Deflated Sharpe ≥0.95 for the portfolio.</p>"
            "<h3>Rank-IC</h3><table><tr><th>Model</th><th>Mean IC</th><th>IC-IR</th>"
            f"<th>t-stat</th><th>n</th><th>verdict</th></tr>{rows}</table>"
            "<h3>Long-short portfolio</h3><table><tr><th>Model</th><th>Sharpe</th>"
            f"<th>Mean net</th><th>DSR</th><th>verdict</th></tr>{pf_rows}</table>"
            f"<p><img src='{os.path.basename(paths['equity_png'])}' width='720'></p>"
            "</body></html>")
    with open(paths["html"], "w") as f:
        f.write(html)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alpha/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alpha/report.py tests/alpha/test_report.py
git commit -m "feat(alpha): IC/portfolio report with equity-curve plot (HTML escaped)"
```

---

### Task 11: CLI + end-to-end integration test + README

**Files:**
- Create: `alpha/cli.py`, `alpha/__main__.py`, `tests/alpha/test_integration.py`, `alpha/README.md`
- Test: `tests/alpha/test_integration.py`

**Interfaces:**
- Produces `alpha.cli.main(argv) -> int` with a `run` subcommand (flags `--universe`, `--horizon`, `--out`, `--cost-bps`): builds the provider via `hub.data.provider.get_default_provider`, builds the panel, runs the backtest, writes the report, prints a plain-text IC/DSR summary. `__main__.py` calls it.
- Integration test drives `run_backtest` + `write_report` over a synthetic panel and asserts the full artifact set, and exercises `main` with `build_panel` monkeypatched (no network).

- [ ] **Step 1: Write the failing test**

```python
# tests/alpha/test_integration.py
import os
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest
from alpha.report import write_report

def test_end_to_end_offline(tmp_path, planted_panel):
    cfg = AlphaConfig(out_dir=str(tmp_path), n_folds=3, purge=1)
    result = run_backtest(planted_panel, cfg)
    paths = write_report(result, planted_panel, cfg, "20260628")
    assert os.path.exists(paths["html"]) and os.path.exists(paths["equity_png"])
    assert result["ic"]["rvol"]["mean_ic"] > 0.05

def test_cli_run_monkeypatched(tmp_path, planted_panel, monkeypatch):
    import alpha.cli as cli
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    monkeypatch.setattr(cli, "build_panel", lambda provider, cfg: planted_panel)
    rc = cli.main(["run", "--out", str(tmp_path)])
    assert rc == 0
    assert any(p.name.startswith("alpha_") for p in tmp_path.iterdir())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alpha/test_integration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# alpha/cli.py
import argparse, datetime as dt
from .config import AlphaConfig
from .panel import build_panel
from .backtest import run_backtest
from .report import write_report
from hub.data.provider import get_default_provider

def main(argv) -> int:
    p = argparse.ArgumentParser(prog="alpha")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--universe"); r.add_argument("--out")
    r.add_argument("--horizon", type=int); r.add_argument("--cost-bps", type=float)
    args = p.parse_args(argv)
    if args.cmd != "run":
        return 1
    cfg = AlphaConfig.default()
    over = {}
    if args.universe: over["universe"] = args.universe
    if args.out: over["out_dir"] = args.out
    if args.horizon: over["horizon"] = args.horizon
    if args.cost_bps is not None: over["cost_bps"] = args.cost_bps
    if over:
        cfg = AlphaConfig(**{**cfg.__dict__, **over})
    provider = get_default_provider(cfg.cache_dir)
    panel = build_panel(provider, cfg)
    if len(panel) == 0:
        print("empty panel (no data fetched)"); return 1
    result = run_backtest(panel, cfg)
    date_str = dt.datetime.now().strftime("%Y%m%d")
    paths = write_report(result, panel, cfg, date_str)
    print(f"panel: {result['n_rows']} rows over {result['n_dates']} rebalances")
    print("Rank-IC (market+sector-neutral, OOS):")
    for name, s in result["ic"].items():
        verdict = "significant" if abs(s["t_stat"]) >= 2 else "noise"
        print(f"  {name:12s} IC={s['mean_ic']:+.4f}  t={s['t_stat']:+.2f}  ({verdict})")
    for name, v in result["portfolio"].items():
        print(f"  [{name}] Sharpe={v['sharpe']:.2f}  DSR={v['dsr']:.2f}  "
              f"({'PASS' if v['dsr']>=0.95 else 'not significant'})")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0
```

```python
# alpha/__main__.py
import sys
from .cli import main
sys.exit(main(sys.argv[1:]))
```

```markdown
# Alpha Engine (Phase A)

Cross-sectional, factor-neutral backtest of the Discovery Hub's signals — the honest
test of whether they hold tradable alpha after neutralization, costs, and
multiple-testing correction.

> ⚠️ Research/measurement harness. Not trading, not advice. Expect IC ≈ 0.02–0.05
> at best on this liquid universe; a bigger number usually means overfitting.

## Run

    .venv/bin/pip install -r alpha/requirements.txt
    .venv/bin/python -m alpha run                 # full live backtest (yfinance)
    .venv/bin/python -m alpha run --cost-bps 5    # override costs

Output in `out_alpha/`: IC table (CSV/JSON), HTML report, equity-curve PNG.
Reports Rank-IC (mean, IC-IR, t-stat) per signal + composite + linear/LightGBM
out-of-sample, and the long-short Sharpe with **Deflated Sharpe Ratio**.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/alpha/ -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add alpha/cli.py alpha/__main__.py tests/alpha/test_integration.py alpha/README.md
git commit -m "feat(alpha): CLI run command + end-to-end integration test + README"
```

---

## Self-Review

**Spec coverage:** panel (§3 → Task 2) ✓; market+sector-neutral target (§4 → Task 3) ✓; rank-IC + t-stat (§4 → Task 4) ✓; purged+embargoed walk-forward (§4 → Task 5) ✓; linear+LightGBM combiner OOS (§3/§5 → Task 6) ✓; long-short portfolio + costs (§4 → Task 7) ✓; Deflated Sharpe (§4 → Task 8) ✓; orchestration reporting every model honestly (§5 → Task 9) ✓; report incl. equity curve (§3 → Task 10) ✓; CLI + integration + README (§3/§9 → Task 11) ✓; error isolation (Task 2/9) ✓; synthetic-panel detect-and-reject tests (§7 → Tasks 1,4,9) ✓; deps pinned (Task 1) ✓. Deferred per spec §10: universe expansion, orthogonal features, FF residualization — not in this plan (correct).

**Placeholder scan:** No TBD/TODO; every code step has real code; data files spelled out in full.

**Type consistency:** Panel schema (`date,ticker,sector,<7 signals>,fwd_ret`) is produced in Task 2 and consumed by Tasks 3,4,6,7,9,10. `neutralize → pd.Series` (Task 3) feeds Tasks 4/6/9 as `.values`. `rank_ic_series(dates,x,y) → Series` + `ic_summary → dict` (Task 4) consumed by Task 9. `purged_walk_forward → list[(train,test)]` (Task 5) consumed by Task 6/9. `composite_score/linear_oos/lgbm_oos → np.ndarray` (Task 6) consumed by Task 9. `long_short → dict` (Task 7) consumed by Task 9/10. `deflated_sharpe_ratio` signature (Task 8) matches its Task 9 call. `run_backtest → dict{ic,portfolio,n_dates,n_rows}` (Task 9) consumed by Task 10/11. Signal names come from `hub.config.SIGNAL_NAMES` everywhere. ✓
