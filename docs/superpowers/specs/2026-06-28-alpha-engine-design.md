# Alpha Engine (Phase A) — Cross-Sectional, Factor-Neutral Backtest Spec

**Date:** 2026-06-28
**Status:** Draft for review
**Branch:** `alpha-engine`
**Builds on:** the merged `hub/` package (reuses its data provider + the 7 signals as features).

## 1. Goal

Answer one honest question: **does the Discovery Hub's signal set contain any tradable cross-sectional alpha, after factor-neutralization, transaction costs, and multiple-testing correction?** Not "does a stock go up" but "does ranking names by our signals beat a market+sector-neutral baseline across hundreds of name-dates."

Why this and not the hub's backtest: the hub measured raw forward returns on 30 names over 12 windows — beta-dominated and statistically empty (edge ≈ +0.2% = noise). This rebuilds the measurement the way a quant desk would: **cross-sectional, factor-neutral target · breadth · purged walk-forward · costs · Deflated Sharpe.**

## 2. Non-goals / honest framing

- Not a live trading system, not order execution, not advice. A research/measurement harness.
- Realistic success = **IC ≈ 0.02–0.05 that is statistically significant after correction**, long-short Sharpe ~0.5–1.5 after costs. A *bigger* number is almost always overfitting — the harness is designed to expose that, not flatter it.
- Phase A universe is a bundled ~100 liquid US names for a buildable MVP; real breadth (500–3000) is config, a fast-follow.
- No paid data: yfinance only (prices + bundled sector map). Fundamentals/alt-data features are a later phase.

## 3. Architecture

New package `alpha/` (sibling of `hub/`), small single-purpose modules. Reuses `hub.data.provider` and `hub.signals.SIGNALS`.

```
alpha/
  config.py        # AlphaConfig: universe, history_days, horizon, n_quantiles, cost_bps, embargo, etc.
  panel.py         # build_panel(provider, cfg) -> long DataFrame [date, ticker, <signal features>, fwd_ret]
  sectors.py       # load_sectors() -> {ticker: sector} from bundled map
  neutralize.py    # market+sector-neutral residual of forward returns, cross-sectionally per date
  ic.py            # rank_ic_series(panel, feature, target) -> per-date Spearman IC; summary (mean, IR, t-stat)
  splits.py        # purged + embargoed walk-forward fold generator over dates
  combine.py       # learn a cross-sectional combination: linear baseline + LightGBM, purged CV -> predictions
  portfolio.py     # long-short top/bottom-quantile sim: turnover, cost_bps, equity curve, Sharpe
  stats.py         # deflated_sharpe_ratio(...), t-stat helpers
  backtest.py      # orchestrate: panel -> neutralize -> IC table -> combine -> portfolio -> metrics
  report.py        # write IC table + equity-curve PNG + metrics (CSV/JSON/HTML)
  cli.py           # `python -m alpha run`
```

## 4. Core definitions (the math that matters)

- **Panel:** rows are (date, ticker). Features = the 7 hub signals computed on each ticker's history up to `date` (point-in-time, no lookahead). Target = `fwd_ret` = ticker's return from `date` to `date+horizon`.
- **Factor-neutral target:** for each date, residualize `fwd_ret` cross-sectionally: subtract the universe mean (market-neutral) and the per-sector mean (sector-neutral). This is the label the engine is scored against — it strips the beta that made the hub's edge ≈ 0.
- **Rank-IC:** per date, Spearman correlation between a feature (or model prediction) and the neutralized target across the cross-section. Report mean IC, IC-IR = mean/std, and t-stat = IC-IR·√(n_dates). This is the primary edge metric.
- **Purged + embargoed walk-forward:** because the target spans `horizon` days, train and test windows that overlap in time leak. Folds purge any training date whose target window overlaps the test window, plus an `embargo` gap. Predictions are strictly out-of-sample.
- **Long-short portfolio:** each rebalance date, long the top-quantile and short the bottom-quantile by the score, equal-weight, market-neutral by construction. Apply `cost_bps` × turnover. Report annualized Sharpe after costs and the equity curve.
- **Deflated Sharpe Ratio (DSR):** corrects the observed Sharpe for the number of strategy configurations tried and non-normal returns (Bailey & López de Prado). A positive raw Sharpe with DSR≈0 means "indistinguishable from luck given how many things we tried."

## 5. What the engine reports (honesty, built in)

For each of: every single signal, the hand-weighted hub composite, and the LightGBM combination — on the **neutralized** target, out-of-sample:
- mean Rank-IC, IC-IR, IC t-stat, n_dates
- long-short annualized return + Sharpe **after costs**, max drawdown
- **Deflated Sharpe Ratio** and whether it clears significance
- a one-line verdict per model (e.g. "IC 0.018, t=1.1, DSR 0.07 → not significant")

The report states plainly when nothing is significant. No window cherry-picking; full sample, fixed config.

## 6. Error handling

- Per-ticker fetch failures isolated (drop the ticker, log it) — never abort the panel.
- A date with too few names (< min_names, default 20) is dropped from IC/portfolio.
- Sectors missing for a ticker → assigned `"UNKNOWN"` sector bucket (still market-neutralized).
- LightGBM/feature NaNs handled (drop or median-fill at the boundary, documented).

## 7. Testing (≥80% on math modules)

- **Synthetic panel with a planted signal:** a feature constructed to predict the neutralized target → IC significantly > 0 and DSR > 0. A pure-noise feature → IC ≈ 0, t < 2, DSR ≈ 0. This proves the harness can *both* detect and reject signal.
- **Neutralization:** after market+sector neutralization, each date's cross-section has ~0 mean and each sector ~0 mean.
- **Purged splits:** assert no train date's target window overlaps any test date (leakage guard).
- **IC:** known monotonic relationship → IC ≈ +1; reversed → −1.
- **Portfolio/costs:** zero-cost vs positive-cost run; higher turnover ⇒ higher cost drag.
- **DSR:** known inputs reproduce the published formula; more trials lower the DSR.
- Synthetic-panel tests need no network; only the live `alpha run` hits yfinance.

## 8. Tech stack / deps

Python 3.13 venv. Reuse pandas, numpy, scipy, matplotlib, yfinance, and `hub/`. Add **lightgbm** and **scikit-learn**. Pinned in `alpha/requirements.txt`.

## 9. Build order (phases within this MVP)

1. Config + synthetic-panel fixtures + sector map + universe list.
2. Panel builder (point-in-time features + forward target).
3. Neutralization.
4. Rank-IC + summary stats.
5. Purged/embargoed walk-forward splits.
6. Linear + LightGBM combiner (purged CV).
7. Long-short portfolio with costs.
8. Deflated Sharpe + significance.
9. Backtest orchestration + report.
10. CLI + integration test + README.

## 10. Open questions / future

- Expand universe to 500–3000 and add orthogonal features (PEAD, analyst revisions, short interest, options skew) — the real edge drivers; needs better data than yfinance.
- Full Fama-French residualization (Ken French factor data) instead of market+sector demeaning.
- Regime conditioning; horizon sweep; capacity/slippage modeling.
