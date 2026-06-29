# Research Tool v2 — Step 4: Polish Spec

**Date:** 2026-06-29
**Status:** Draft (autonomous build)
**Branch:** `step4-polish`
**Builds on:** the merged hub + OpenBB backbone + screening + Streamlit UI.

## 1. Goal

Two high-value polish features:
1. **Deep "why" from SEC filings** — feed real 10-K risk-factors / business / MD&A text (via `edgartools`) into the LLM analysis, so the per-candidate "why" is grounded in actual disclosures, not just price + metrics.
2. **UI usability** — CSV export of the watchlist + saved screens (name → persisted slider/preset config you can reload).

(Deferred, with reason: quantstats tearsheets need a historical screen-backtest = a bigger feature; a Kronos forecast overlay would sit awkwardly against our proven finding that Kronos doesn't predict — adding it as a "signal" risks misleading.)

## 2. Non-goals

- No alpha claim — filings make the *research* deeper, not the picks predictive.
- Not a full filings browser; just the latest 10-K's key sections, truncated, for the LLM.

## 3. Architecture

```
hub/data/filings.py    # FilingProvider: cached latest-10-K section text (risk_factors/business/mda),
                       #   configurable SEC identity, injectable edgar client for offline tests
hub/explain.py         # optional filing_provider -> filing excerpts added to the LLM prompt
hub/cli.py + hub/ui/app.py  # construct a FilingProvider when LLM is enabled; pass it to explain_top
hub/ui/screen_runner.py     # save_screen / load_screens helpers (JSON persistence)
hub/ui/app.py          # sidebar: save/load screen; main: CSV download_button for the watchlist
```

## 4. Filings contract

- `FilingProvider(identity=None, kv=None, edgar=None)` — `identity` defaults to env `SEC_IDENTITY` or a generic `"kronos-research-tool research@example.com"` (SEC requires *some* contact User-Agent; we do NOT hardcode the user's personal email — it's configurable).
- `get_filing_summary(symbol, max_chars=2000) -> dict`: `{form, date, sections: {risk_factors?, business?, mda?}}`, each section whitespace-collapsed and truncated to `max_chars`. Cached (filings rarely change → long TTL). Any edgar error → `{form:None, date:None, sections:{}}` (never raises).
- `edgar` injectable: a stand-in exposing `Company(symbol).get_filings(form=...).latest(1).obj().<section>` for offline tests.

## 5. Deep "why" wiring

- `explain_top(result, provider, client, cfg, filing_provider=None)` — backward compatible (None ⇒ Step-1 behavior). When provided, `explain_candidate` appends a condensed filing excerpt (risk factors first, then business) to the prompt and asks the LLM to ground the bull/bear/risk in the disclosures.
- The CLI (`scan`/`screen`) and the UI construct a `FilingProvider` only when the LLM step runs.

## 6. Saved screens + CSV

- `save_screen(name, settings, path)` / `load_screens(path) -> dict[name, settings]` in `screen_runner.py` — JSON file persistence; corrupt/missing → `{}`; testable offline.
- UI: a sidebar "Save current screen" (name input + button) and a "Load saved screen" selectbox that pre-fills the controls; a "Download watchlist (CSV)" button on the table.

## 7. Error handling

- edgar unreachable / no 10-K / parse failure → empty sections; the "why" degrades to news+fundamentals (still works).
- Saved-screens file unreadable → treated as empty; saving failures are swallowed (don't crash the UI).

## 8. Testing (no network)

- `FilingProvider.get_filing_summary` with an injected fake edgar: sections extracted + truncated to `max_chars`; missing section omitted; error → empty dict, no crash; cache round-trip.
- `explain_candidate` includes filing text in the prompt when a filing_provider yields sections (stub client captures the prompt).
- `save_screen`/`load_screens` round-trip + corrupt-file safety.
- Live smoke: real edgar fetch for one ticker returns non-empty risk_factors; UI re-verified by screenshot (CSV button + save/load present).

## 9. Tech stack / deps

Add `edgartools` to `hub/requirements.txt` (verified compatible: pandas≥2.0, Py3.13).

## 10. Build order

1. `FilingProvider` (edgartools, cached, injectable).
2. `explain.py` filing wiring + CLI/UI hookup.
3. `screen_runner` save/load helpers + UI CSV + saved-screens controls.

## 11. Future (still deferred)

- quantstats tearsheet for a historical screen-backtest; custom criteria builder in the UI; Kronos overlay (only if framed honestly as a non-predictive scenario).
