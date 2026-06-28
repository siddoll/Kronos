# Research Tool v2 — Step 3: Streamlit UI Spec

**Date:** 2026-06-29
**Status:** Draft (autonomous build)
**Branch:** `streamlit-ui`
**Builds on:** the merged hub + OpenBB backbone (Step 1) + screening engine (Step 2).

## 1. Goal

A clickable **Streamlit dashboard** over the screening engine: pick a preset thesis, tune key thresholds with sliders, run the screen, browse the ranked watchlist, and drill into any name (candlestick chart + fundamentals + criteria ✓/✗ + optional LLM "why"). This is the milestone where the CLI tool becomes a *tool you actually use*.

streamlit 1.58.0 + plotly 6.8.0 verified installed.

## 2. Non-goals

- No new screening logic — pure presentation over Step 2's `run_screen`. No alpha claim (a screen is a filter+ranking for human judgment).
- Not a hosted/multi-user app; a local single-user dashboard (`streamlit run`).

## 3. Architecture

```
hub/ui/
  screen_runner.py  # TESTABLE seam: build_criteria(preset, overrides) -> [Criterion];
                    #   screen_to_table(result) -> DataFrame; PRESET_NAMES
  app.py            # Streamlit app (thin presentation): sidebar controls -> run_screen
                    #   (cached) -> ranked table -> per-symbol detail (plotly chart + funds + criteria + LLM)
  run.py            # `python -m hub.ui` -> launches `streamlit run hub/ui/app.py`
```

Logic lives in `screen_runner.py` (unit-tested, no Streamlit/network); `app.py` is verified by launching the server and screenshotting it.

## 4. UX

- **Sidebar:** universe select, preset select, sliders (Max P/E, Min earnings growth, Within % of 52w-high, Top-K), an "include LLM analysis" toggle (off by default — costs API), and a **Run** button.
- **Main:** "{N} matches" + a sortable ranked table (Symbol, Score, Criteria N/M, P/E, EPS growth, Net margin); a symbol picker → detail: 1-year **candlestick** (plotly), fundamentals (the non-null OpenBB metrics), criteria ✓/✗ with values, and the LLM "why" when enabled.
- Caching: provider via `@st.cache_resource`; the screen run via `@st.cache_data` keyed by the control values (so re-runs are instant unless settings change). Honest caption: "Research tool, not buy signals."

## 5. screen_runner contract

- `PRESET_NAMES: list[str]` = the preset keys.
- `build_criteria(preset: str, overrides: dict) -> list[Criterion]`: starts from `get_preset(preset)`; where a criterion's threshold is overridden (`pe_max` → pe_below, `eps_growth_min` → earnings-growth, `near_high_pct` → near_52w_high), rebuild that criterion with the slider value; others unchanged. Unknown overrides for a preset that lacks them are ignored.
- `screen_to_table(result: dict) -> pandas.DataFrame`: one row per candidate — Symbol, Score, "passed/total" criteria, P/E, EPS growth, Net margin (None-safe).

## 6. Error handling

- Empty result → table is empty, a friendly "no matches — loosen the filters" message, no detail panel.
- A per-symbol chart/fetch failure in the detail view → an inline warning, the rest of the page still renders.
- LLM toggle off → no Anthropic import/call.

## 7. Testing

- `build_criteria`: overriding `pe_max` yields a pe criterion that fails a higher P/E and passes a lower one; an override irrelevant to the preset is ignored; returns `[Criterion]`.
- `screen_to_table`: maps a synthetic `run_screen`-shaped result to the expected columns; None fundamentals render safely.
- The Streamlit `app.py` is verified by **launching it headless and screenshotting** (a real frame with the controls + table = pass; a blank/error frame = fail) — not unit-tested.

## 8. Tech stack / deps

Add `streamlit` + (already-present) `plotly` to `hub/requirements.txt`.

## 9. Build order

1. `screen_runner.py` (build_criteria, screen_to_table, PRESET_NAMES) + tests.
2. `app.py` + `run.py` launcher + requirements; launch + screenshot verification.

## 10. Future

- Custom criteria builder (add/remove any criterion in the UI), saved screens, CSV export of the watchlist, edgartools filing text in the detail "why", Kronos forecast overlay on the chart.
