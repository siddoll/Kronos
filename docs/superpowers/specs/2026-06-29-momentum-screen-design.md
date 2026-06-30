# Momentum/Catalyst Screen + Honest Forward-Test Spec

**Date:** 2026-06-29
**Status:** Draft (autonomous build)
**Branch:** `momentum-screen`
**Builds on:** the merged hub screener + UI.

## 1. Goal

The user wants "stocks that will probably rise in 1–4 weeks." That exact prediction was *empirically disproven* this project (SPCX, Phase A, the breadth experiment). So we build the **honest** version:
1. A **momentum/strength screen** that surfaces names *currently exhibiting* the price patterns that often precede short-term moves (relative-volume spike, near 52-week high, positive short-term momentum, uptrend) — labeled "showing strength now," **not** "will rise."
2. A built-in **forward-test** that, for this exact screen's price criteria, backtests over years and reports the *real* hit-rate and forward return of its picks vs the market at 1/2/4-week horizons — so the user sees the truth (expected ~coin-flip, no significant edge) instead of trusting a false promise.

The integrity feature is the forward-test: it must be impossible to use the screen without seeing how (un)reliable it is.

## 2. Non-goals / honesty constraints

- **No "will rise" label, no probability-of-gain claim.** The screen ranks by a strength profile; it does not predict.
- The forward-test uses ONLY **point-in-time price criteria** (computable from historical prices without lookahead). Fundamental/catalyst filters use snapshot data → cannot be honestly backtested → they are NOT in the forward-test (noted in the UI). The live screen may still rank with them; the LLM "why" surfaces catalysts.
- The forward-test reports results plainly even when (especially when) there is no edge.

## 3. Architecture (extends `hub/screen/` + the UI)

```
hub/screen/criteria.py   # + rvol_above(mult), short_momentum_positive(window)
hub/screen/library.py    # + PRESETS["momentum_catalyst"] (price-strength profile)
hub/screen/forward_test.py  # forward_test(frames, criteria, horizons) -> per-horizon
                            #   hit_rate / pick_return / market_return / edge / n  (price criteria only)
hub/ui/app.py            # a prominent "Reality check" panel: runs the forward-test for the
                         #   active screen and shows the honest hit-rate table + verdict
```

## 4. New criteria

- `rvol_above(mult=1.5, window=20, hard=False)` — today's volume / avg(volume, window); soft score scales with the spike.
- `short_momentum_positive(window=20, hard=True)` — `close[-1]/close[-window] - 1`; the recent-strength leg.

## 5. The momentum_catalyst preset

`[rvol_above(1.5, hard=False), near_52w_high(0.12), above_sma(50), short_momentum_positive(20)]` — names with a volume spike, near their high, above trend, with positive recent momentum. All price-based ⇒ fully backtestable.

## 6. Forward-test contract

`forward_test(frames, criteria, horizons=(5,10,20), step=5, warmup=252) -> dict`:
- Use only `kind=="technical"` criteria (point-in-time). For each rebalance date (every `step` bars from `warmup`), evaluate the hard technical filters on each ticker's history-up-to-that-bar; the survivors are that date's "picks". For each horizon `h`: pick forward return = `close[d+h]/close[d]-1`, market forward return = mean over all names with data.
- Aggregate per horizon: `hit_rate` (% picks with positive forward return), `pick_return` (mean), `market_return` (mean), `edge` (pick−market), `n` (pick-instances).
- Returns `{"horizons": {h: {...}}, "n_dates": int, "n_names": int}`. Empty/insufficient → zeros.

## 7. UI "Reality check" panel

Below the watchlist, a clearly-headed expander **"⚖️ Reality check — how reliable is this screen?"** that runs `forward_test` on the active screen's price criteria over the loaded universe and shows a table (1wk/2wk/4wk → hit-rate, avg pick return, avg market, edge, n) plus a bold honest caption, e.g.: *"Historically the picks rose ~X% of the time vs ~Y% for the market — an edge of Z. This is not a prediction; treat it as a starting point, not a buy signal."* If the edge is tiny/insignificant, say so explicitly.

## 8. Error handling

- A name with insufficient history at a date → skipped for that date (no crash).
- A criterion erroring → that name fails (fail-safe, reusing CritResult behavior).
- Empty universe / no picks → the panel shows "not enough data" rather than fake numbers.

## 9. Testing (no network)

- `rvol_above` / `short_momentum_positive`: pass/fail + score on synthetic price; insufficient history → fail-safe.
- `forward_test` on a synthetic set where one name has a planted persistent uptrend and another is flat → picks (the trending one) have higher hit-rate than the flat; structure/keys correct; a no-data case → zeros, no crash. Crucially also: a **random-walk synthetic universe → hit_rate ≈ 0.5 and edge ≈ 0** (the harness must NOT manufacture an edge).
- UI panel verified by launch + screenshot.

## 10. Build order

1. New criteria (`rvol_above`, `short_momentum_positive`) + `momentum_catalyst` preset.
2. `forward_test.py` (price-criteria screen backtest, multi-horizon).
3. UI "Reality check" panel + live verification.

## 11. Future

- A point-in-time fundamentals/catalyst feed (paid) would let the forward-test include the catalyst leg honestly.
- Per-criterion contribution breakdown; turnover/cost-adjusted forward returns.
