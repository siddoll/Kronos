# Research Tool v2 — Step 2: Configurable Screening Spec

**Date:** 2026-06-28
**Status:** Draft (autonomous build)
**Branch:** `screening`
**Builds on:** the merged `hub/` + OpenBB backbone (Step 1). Goal B = a usable research/signal tool for finding stock picks.

## 1. Goal

Turn the hub into a real **stock screener**: the user expresses a thesis as a set of **criteria** — technical (via `pandas-ta`) AND fundamental (via the OpenBB backbone) — and the tool returns the names that pass, ranked, with the LLM "why". E.g. *"near 52-week high AND positive 12-1 momentum AND earnings growth > 10% AND P/E < 40."* This is the heart of "find good picks."

`pandas-ta` 0.4.71b0 is verified working on this venv (Python 3.13, pandas 3.0).

## 2. Non-goals (this step)

- No alpha claim — a screen is a *configurable filter + ranking*, not a prediction (the breadth experiment proved price signals don't forecast returns). It surfaces candidates matching the user's thesis for human judgment.
- Not the Streamlit UI (Step 3) — this step is the screening engine + CLI. Not edgartools/quantstats (Step 4).

## 3. Architecture (new `hub/screen/` package; reuses the provider + report + explain)

```
hub/screen/
  criteria.py    # Criterion dataclass: name, kind(technical|fundamental), hard(bool),
                 #   evaluate(price_df, fundamentals) -> CritResult(passed, score in [0,1], value)
  library.py     # factory functions for ready criteria + PRESETS (named criteria bundles)
  screener.py    # run_screen(universe, provider, criteria, top_k) -> {candidates, skipped}
                 #   STAGED: technical hard-filters first (price only, cheap) -> survivors fetch
                 #   fundamentals (cached) -> fundamental hard-filters -> soft-score -> rank
  cli is hub/cli.py: a `screen` subcommand (--preset, --top-k, --no-explain)
```

The staged design keeps it fast: fundamentals (one OpenBB call/ticker) are fetched only for the names that already pass the technical filters, not the whole universe.

## 4. The criterion model

- **CritResult**: `passed: bool`, `score: float in [0,1]`, `value: float` (the raw measured value, for the report).
- **Criterion**: `name`, `kind` ∈ {"technical","fundamental"}, `hard` (True ⇒ failing drops the name; False ⇒ contributes only to the soft score), `evaluate(price_df, fundamentals) -> CritResult`.
- **Technical criteria** (from price via pandas-ta / pandas): `near_52w_high(within_pct)`, `momentum_12_1_positive()`, `above_sma(n)`, `rsi_between(lo, hi)`, `macd_bullish()`, `adx_above(n)`.
- **Fundamental criteria** (threshold on an OpenBB fundamentals key): `pe_below(x)`, `peg_below(x)`, `growth_above(key, x)` (earnings_growth/revenue_growth), `margin_above(key, x)` (gross/net), `mktcap_above(x)`.
- **Composite score** = mean of all criteria's soft scores (hard + soft). Names passing ALL hard filters are ranked by this; top-K returned.

## 5. Presets (named thesis bundles, user-extensible)

- `growth_momentum`: near 52w-high + positive 12-1 momentum + above SMA200 + earnings_growth>10% + P/E<40.
- `value`: P/E<15 + net_margin>10% + revenue_growth>0 + mktcap>2e9.
- `quality_momentum`: above SMA200 + RSI 50-70 + net_margin>15% + earnings_growth>0.

## 6. What the tool gains / output

- CLI `hub screen --preset growth_momentum` → ranked watchlist of names passing the thesis, each with: composite score, **which criteria passed** (✓/✗ + value), key fundamentals, and the LLM "why" for the top-K.
- Report (HTML/CSV) extended with a per-candidate criteria-pass summary.

## 7. Error handling

- Per-ticker price/fundamentals fetch failure → ticker skipped (logged), never aborts the screen.
- A criterion that errors on a name (e.g. insufficient history) → treated as `passed=False, score=0` (does not crash the screen).
- Missing fundamental (None) → fundamental criterion fails closed (passed=False) so a hard fundamental filter never passes a name with unknown data.

## 8. Testing (no network)

- Each criterion on synthetic price/fundamentals: a name that should pass passes; one that shouldn't fails; score in [0,1]; insufficient-data → fail not crash.
- Staged screener with a stub provider: technical hard-filter drops names before any fundamental fetch (assert the stub's fundamentals are only requested for technical survivors); fundamental hard-filter drops; ranking orders by composite; top-K respected.
- A preset end-to-end on synthetic data → produces a ranked, criteria-annotated result.

## 9. Tech stack / deps

Add `pandas-ta` (0.4.71b0, verified) to `hub/requirements.txt`. Reuse OpenBB provider, report, explain.

## 10. Build order

1. Criterion model + technical criteria (pandas-ta).
2. Fundamental criteria + library + presets.
3. Staged screener (technical→fundamental→rank) with fetch-minimization.
4. CLI `screen` + report criteria column + integration test + live smoke (real OpenBB on a small universe).

## 11. Open questions / future

- Step 3: Streamlit UI exposing the criteria as interactive controls (sliders/toggles) over this engine.
- Custom user criteria from a config file/JSON; saved screens.
- edgartools filing-text criteria; quantstats tearsheet for a screen's historical hit-rate.
