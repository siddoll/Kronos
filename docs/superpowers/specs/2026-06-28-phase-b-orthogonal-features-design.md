# Phase B — Orthogonal (Non-Price) Features Spec

**Date:** 2026-06-28
**Status:** Draft (autonomous build — design decisions noted inline)
**Branch:** `phase-b-features`
**Builds on:** the merged `alpha/` engine (Phase A). Reuses its panel, neutralization, IC, splits, combiners, portfolio, DSR, report.

## 1. Goal

Test whether adding **orthogonal, non-price features** to the alpha panel produces measurable cross-sectional alpha that the price signals alone did not (Phase A verdict: price signals → IC ≈ noise, composite DSR 0.39). Same honest bar: market+sector-neutral, purged walk-forward, costs, Deflated Sharpe.

## 2. Data-availability reality (decided during a live yfinance probe)

The user named PEAD, analyst revisions, and short interest. What is actually buildable **point-in-time** from free yfinance:

| Source | yfinance | Verdict |
|---|---|---|
| **Analyst revisions** | `Ticker.upgrades_downgrades` — dated history (`GradeDate`, grade action, price-target raises/lowers) | ✅ buildable, point-in-time-safe |
| **Earnings / PEAD** | `Ticker.get_earnings_dates()` — dated history with `Surprise(%)` (needs `lxml`) | ✅ buildable, point-in-time-safe |
| **Short interest** | `Ticker.info` — current **snapshot only**, no history | ❌ not usable historically (lookahead). **Documented paid-data gap**; provider hook left ready. |

So Phase B adds **two** orthogonal features — `rev_mom` (revision momentum) and `pead` (earnings drift) — and documents short interest as needing a paid feed.

> Honest caveat: yfinance revision/earnings coverage and history depth vary by name and may be thin/restated. The harness measures IC regardless; if the data is too sparse, IC will be ~0 and we report that plainly.

## 3. Architecture (extends `alpha/`, does not touch `hub/`)

```
alpha/
  data.py            # ExternalDataProvider: cached get_upgrades_downgrades(t), get_earnings(t)
  features/
    __init__.py
    revisions.py     # revision_momentum(ud_df, as_of_dates, window_days) -> {date: float}
    earnings.py      # earnings_drift(earn_df, as_of_dates, decay_days) -> {date: float}
  panel.py           # build_panel(..., ext_provider=None): adds rev_mom, pead columns when given
  config.py          # + extra_features: tuple = (); include_orthogonal flag path
  backtest.py        # feats = SIGNAL_NAMES + extra_features; also reports price-only vs price+orthogonal OOS combiner
  cli.py             # `alpha run --with-orthogonal`
```

## 4. Feature definitions (point-in-time, no lookahead)

- **`rev_mom`** — for each (ticker, rebalance date `d`): over revisions with `GradeDate ∈ (d − window_days, d]`, score = (#upgrades − #downgrades) where an upgrade = grade action up OR price-target raised, downgrade = grade action down OR price-target lowered. Default `window_days = 90`. No data in window → NaN.
- **`pead`** — most recent earnings with `date ≤ d`; feature = `Surprise(%) × exp(−days_since / decay_days)` (drift decays after the announcement). Default `decay_days = 60`. No prior earnings → NaN.
- Both use ONLY information dated `≤ d`. NaN features are median-filled per date inside the combiner (documented) so a missing orthogonal value never drops a name.

## 5. What the run reports

Same engine, but `feats = the 7 price signals + rev_mom + pead`. Report:
- Rank-IC (+t-stat) for **every** feature individually — including `rev_mom`, `pead`.
- **The key comparison:** out-of-sample LightGBM (and linear) IC on **price-only** vs **price+orthogonal**. Does adding the orthogonal features raise OOS IC, and does either clear significance (|t|≥2 / DSR≥0.95)?
- Long-short Sharpe + Deflated Sharpe for the price+orthogonal composite/combiner.
- Plain verdict, including "the orthogonal features add nothing measurable" if that's the truth.

## 6. Error handling

- Per-ticker external-data fetch failure → that ticker's `rev_mom`/`pead` are NaN, never aborts the panel.
- NaN orthogonal features median-filled per date in the combiner; if an entire feature is empty, it contributes ~0 and is reported as such.
- Backward compatible: with `ext_provider=None` / `extra_features=()`, `build_panel` and `run_backtest` behave exactly as Phase A (existing tests stay green).

## 7. Testing (≥80% on feature math)

- **Point-in-time correctness:** synthetic upgrades/earnings frames where a *future* revision/earnings must NOT affect an earlier date's feature (leakage guard).
- **revision_momentum:** more upgrades than downgrades in window → positive; symmetric → ~0; empty window → NaN.
- **earnings_drift:** positive surprise → positive, decaying with days-since; no prior earnings → NaN.
- **Synthetic planted-orthogonal panel:** a panel where `rev_mom` (not the price signals) predicts the neutralized target → the price+orthogonal combiner's OOS IC beats the price-only combiner. Proves the comparison can detect added value.
- Live external data not required for unit tests (synthetic frames); only the live `alpha run --with-orthogonal` touches yfinance.

## 8. Tech stack / deps

Reuse `alpha/` + add `lxml` (required by yfinance `get_earnings_dates`). Pinned in `alpha/requirements.txt`.

## 9. Build order

1. ExternalDataProvider (cached yfinance revisions + earnings) + lxml dep.
2. revision_momentum + earnings_drift feature builders (point-in-time).
3. Panel extension (ext_provider → rev_mom, pead columns).
4. Config + backtest extension (extra_features; price-only vs price+orthogonal OOS comparison).
5. CLI `--with-orthogonal` + report extension + integration test + README + live run.

## 10. Open questions / future

- True short-interest history, analyst EPS-estimate revisions (vs grade changes), institutional ownership, alt-data → all need paid feeds; the `ExternalDataProvider` interface is the seam to add them.
- If orthogonal features show standalone IC but the universe is too small for significance, expand to 500+ names (the Phase A breadth point).
