# Research Tool v2 — Step 1: OpenBB Data Backbone Spec

**Date:** 2026-06-28
**Status:** Draft (autonomous build)
**Branch:** `openbb-backbone`
**Builds on:** the merged `hub/` Discovery Hub. Goal B = a usable research/signal tool for finding stock picks.

## 1. Goal

Replace the coarse yfinance data with **OpenBB** as the hub's data backbone, so the tool screens and explains stocks on **rich fundamentals + real news** — not just price. The LLM "why" and the report immediately get better because they're fed real P/E, growth, margins, ratios, and company news instead of yfinance's sparse `.info`.

OpenBB is verified installed in this venv (Python 3.13, pandas 3.0) with **no dependency downgrades**; `obb.equity.fundamental.metrics`, `obb.news.company`, and `obb.equity.price.historical` all return usable data.

## 2. Non-goals (this step)

- Not the fundamental *screening* logic yet (that's Step 2 with pandas-ta + fundamental filters), not the Streamlit UI (Step 3), not edgartools/quantstats (Step 4). This step is the **data layer** only.
- Not a paid-data integration; uses OpenBB's free providers (yfinance/SEC/FMP-free). No alpha claims — this enriches research, it does not predict.

## 3. Architecture (extends `hub/data/`, behind the existing `DataProvider` interface)

```
hub/data/
  provider.py      # OpenBBProvider: rich get_fundamentals + get_news (was yfinance-delegated);
                   #   dependency-injectable `obb` for testability; robust None/empty fallbacks
  kvcache.py       # tiny JSON KV cache (TTL) for fundamentals/news dicts (parquet cache is df-only)
hub/report.py      # surface key fundamentals (P/E, growth, margin) in the HTML watchlist
```

`get_default_provider` already prefers OpenBB when installed; the hub CLI, `explain.py` (LLM "why"), and `report.py` consume the provider interface unchanged, so they inherit the richer data automatically.

## 4. Provider contract (stable, normalized schemas)

- **`get_fundamentals(symbol) -> dict`** — normalized from `obb.equity.fundamental.metrics(symbol).to_dataframe()`. Fixed keys, float-or-None values: `market_cap, pe_ratio, forward_pe, peg_ratio, earnings_growth, revenue_growth, gross_margin, net_margin, debt_to_equity, current_ratio, dividend_yield`. Missing metric → `None`. Cached (KV, 24h TTL). Any OpenBB error → all-None dict (never crash).
- **`get_news(symbol, limit=5) -> list[dict]`** — from `obb.news.company(symbol, limit=limit).to_dataframe()`, each `{date, title, source}`. Cached. Error/empty → `[]`.
- **`get_ohlcv(symbol, lookback_days) -> DataFrame`** — unchanged OpenBB path (already works; cached via the existing `CachedProvider`/`OHLCVCache`).
- **Testability:** `OpenBBProvider(obb=None)` — if `obb` is None, lazily import the real client; tests inject a fake `obb` exposing the same call surface, so unit tests need NO network.

## 5. What the tool gains

- The LLM analysis (`explain.py`) now reasons over real fundamentals + news → genuinely useful "why this stock / catalyst / risk."
- The HTML report shows each candidate's key fundamentals (P/E, growth, margin) alongside the technical score.
- The data foundation for Step 2's fundamental screening (P/E, growth, margin filters).

## 6. Error handling

- Per-symbol OpenBB failure (timeout, missing data, provider error) is isolated → returns the safe empty value (all-None dict / `[]`); the scan never aborts.
- Normalization tolerates schema variation: read columns by name with `.get`, coerce to float, default None.
- KV cache corruption / missing → treated as a miss.

## 7. Testing (no network)

- Fake-`obb` injection: `get_fundamentals` maps a metrics DataFrame to the normalized dict (known columns → values; missing column → None); error path → all-None.
- `get_news` maps a news DataFrame → list of dicts; empty → `[]`.
- KV cache: put/get round-trip + TTL expiry.
- Report: HTML includes a fundamentals column when candidates carry fundamentals.
- A `@pytest.mark.network` smoke test (skipped by default) that hits real OpenBB for one ticker.

## 8. Tech stack / deps

`openbb` (installed, no downgrades). Pin `openbb` in `hub/requirements.txt`. Everything else reused.

## 9. Build order

1. KV cache (JSON, TTL).
2. OpenBBProvider rich `get_fundamentals` (injectable obb, normalization, cache, fallback).
3. OpenBBProvider `get_news` + wire fundamentals caching into `get_default_provider`.
4. Surface fundamentals in the report + integration test + live smoke (real OpenBB).

## 10. Open questions / future

- Step 2: pandas-ta richer technical signals + **fundamental screens** (filter by P/E/growth/margin) using this data.
- Step 3: Streamlit UI. Step 4: edgartools (filings) for deep "why" + quantstats tearsheets.
- OpenBB premium providers (FMP/Intrinio/Tiingo paid keys) drop in via the same `provider=` arg.
