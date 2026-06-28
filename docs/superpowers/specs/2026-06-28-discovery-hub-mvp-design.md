# Discovery Hub — MVP Design Spec

**Date:** 2026-06-28
**Status:** Draft for review
**Branch:** `discovery-hub`

## 1. Goal

A research **funnel** that scans a stock universe daily, surfaces names showing
**early-mover signals** (the kind that *often precede* multi-day/week swings), ranks
them, and for the top candidates uses an LLM to explain the likely *why* (catalyst,
bull/bear, risk flags). Output is a ranked, explained watchlist.

Timeframe target: **swing (days–weeks), catalyst-driven** (decided in brainstorming).

## 2. Non-goals / honest framing (read first)

- **This is not a predictor / crystal ball.** No model — Kronos included — reliably
  beats a naive baseline on forward *returns* (see prior research + SPCX head-to-head).
  The hub raises hit-rate and saves research time; it does **not** reliably predict surges.
- It surfaces **candidates with false positives**, to be reviewed — not buy signals.
- No order execution, no portfolio management, no financial advice. Research output only.
- MVP universe is **US large/mid + liquid small-caps, free data**. Micro-caps, crypto,
  EU, and paid feeds are explicit future extensions, not in this spec.

## 3. Architecture

UI-agnostic **engine** with a thin surface on top. MVP surface = **CLI/batch**; a
Streamlit dashboard and/or OpenBB Workspace widget come later without engine rework.
Many small, single-purpose modules (per repo coding-style: high cohesion, <400 lines each).

```
hub/
  universe.py     # build scan list (S&P500 + Nasdaq-100 constituents via OpenBB); config-driven
  data/
    provider.py   # Repository interface: get_ohlcv(), get_fundamentals(), get_news()
    openbb_provider.py  # OpenBB Platform impl (free providers: yfinance etc.)
    cache.py      # local parquet/sqlite cache; TTL; per-ticker isolation
  signals/
    base.py       # Signal protocol: name, compute(df) -> float in [0,1]
    rvol.py, breakout.py, trend.py, vcp.py, rsi.py, rangeexp.py, relstrength.py
  rank.py         # weighted composite of sub-scores -> ranked top-K
  explain.py      # LLM (Haiku 4.5 default) writes "why/catalyst/risk" for top-K only
  report.py       # ranked watchlist -> CSV/JSON + Markdown/HTML report
  validate.py     # walk-forward backtest of the screen vs universe/random baseline
  config.py       # universe, weights, top_k, model ids, thresholds
  run.py          # orchestrates: universe -> data -> signals -> rank -> explain -> report
  cli.py          # `python -m hub` entrypoint (scan, backtest subcommands)
```

**Data flow:** `universe → data(OpenBB, cached) → per-ticker signals → composite rank → top-K → LLM explain → report`.

## 4. Screening signals (the core)

Each signal is a **pure function** over a single ticker's OHLCV (+ optional fundamentals)
returning a normalized **0–1 sub-score**. Chosen for the swing/early-mover thesis:

| Signal | Idea | Rough computation |
|---|---|---|
| **RVOL** | Volume spike = fresh interest | `vol_today / mean(vol, 20d)`, squashed (e.g. `min(x/2, 1)`) |
| **Breakout proximity** | Near/breaking N-day high | position in Donchian channel: `(close - low_N) / (high_N - low_N)`, N∈{20,55} |
| **Trend** | Constructive uptrend | price > rising 50-day MA, and MA20 > MA50 → graded score |
| **Volatility contraction (VCP-ish)** | Coiling before a move | `1 - ATR(short)/ATR(long)` clipped to [0,1] |
| **RSI zone** | Momentum, not exhausted | RSI(14): peak score ~55–68, taper toward 0 above ~80 |
| **Range expansion / gap** | Recent thrust | recent gap-up or daily range vs 20d avg range |
| **Relative strength** | Outperforming the index | stock return − SPY return over N days, squashed |

**Composite score** = weighted sum of sub-scores (weights in `config.py`; start tilted to
RVOL + breakout + relative strength — the classic early-mover tells). Rank → **top-K (default 25)**.
All thresholds/weights are config values, never hardcoded magic numbers.

## 5. LLM explanation layer (top-K only)

The **only** LLM cost. For each of the top-K, pull recent news headlines, next earnings
date, and a few fundamentals via OpenBB, then **Haiku 4.5** writes a compact note:
likely catalyst · bull case · bear case · **risk flags** (earnings imminent, low float,
recent dilution, possible pump-and-dump). Provider-abstracted so a cheaper bulk worker
(e.g. Gemini Flash) can be swapped in later. Prompt-cache the shared instruction prefix.
Model ids configurable; default Haiku 4.5, escalate to Sonnet 4.6 for deeper analysis on demand.

## 6. Output / surface (MVP = CLI)

- `python -m hub scan` → writes `out/watchlist_<date>.{csv,json}` + `out/report_<date>.html`
  (ranked table: ticker, composite score, sub-scores, LLM note, risk flags).
- `python -m hub backtest` → runs §7 and prints/saves the validation report.
- Cron-friendly (e.g. daily pre-market). Streamlit dashboard = next sub-project.

## 7. Validation (honesty, built in)

Reuse the existing walk-forward harness pattern. For historical dates: compute the screen,
then measure **forward N-day return of top-K vs (a) universe average and (b) random K**.
Report hit-rate, mean/median forward return, and IC. State results plainly — expected to be
modest; the funnel's value is triage, not alpha. No silent cherry-picking of windows.

## 8. Error handling

- Per-ticker fetch failure is isolated (one bad ticker never kills the scan); logged with reason.
- Missing/insufficient data → ticker skipped, recorded in report's "skipped" section.
- LLM failure → fall back to signal-only output (no explanation), scan still completes.
- All external data validated at the boundary (columns, NaNs, min history length).

## 9. Testing

- **Unit:** each signal function against small fixtures (known input → expected score range).
- **Integration:** full pipeline on a tiny cached universe (~10 tickers), asserts a ranked
  watchlist + report are produced.
- **Validation test:** backtest runs end-to-end on cached history and emits metrics.
- Target ≥80% coverage on `signals/`, `rank.py`, `data/` per repo testing rules.

## 10. Tech stack / dependencies

Python 3.13 venv (existing). Add: `openbb` (4.7.x). Reuse: `pandas`, `numpy`, `scipy`,
`anthropic` (LLM), existing Kronos for optional forecast sub-score later. New deps pinned in
a `hub/requirements.txt`.

## 11. Build order (phases within this MVP)

1. Data layer + cache + universe (provider-agnostic, OpenBB free).
2. Signals + composite ranking (pure, unit-tested) → CLI `scan` producing signal-only report.
3. LLM explanation layer for top-K.
4. Validation/backtest command.
5. (Next sub-project) Streamlit dashboard; later OpenBB Workspace widget / paid feeds / Kronos sub-score.

## 12. Open questions / future

- Add **Kronos forecast** as an optional sub-score once §1–4 prove out (it's available, but
  weak signal — keep it optional and clearly weighted low).
- Paid data feeds (LSEG/S&P/Polygon) behind the same provider interface.
- Expand universe to small/micro-cap and crypto (separate specs; different data quality).
