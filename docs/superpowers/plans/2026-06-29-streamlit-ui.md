# Streamlit UI — Implementation Plan (Research Tool v2, Step 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** A Streamlit dashboard over the screening engine — preset + slider controls → ranked watchlist → per-symbol detail (chart + fundamentals + criteria + LLM). Logic in a testable `screen_runner`; the app verified by launch + screenshot.

**Architecture:** `hub/ui/screen_runner.py` (testable build_criteria/screen_to_table), `hub/ui/app.py` (Streamlit presentation over `run_screen`), `hub/ui/run.py` (launcher).

**Tech Stack:** Python 3.13, streamlit 1.58, plotly 6.8, the merged hub screening engine.

## Global Constraints

- `.venv/bin/python` / `.venv/bin/pytest`. Branch `streamlit-ui`. Add `streamlit` to `hub/requirements.txt`.
- `screen_runner.py` is pure logic — NO streamlit import, NO network — so it's unit-tested offline. `app.py` imports streamlit and is verified by running, not unit-tested.
- Criterion names to match in `build_criteria`: `pe_below` (from `pe_below`), `earnings_growth_above` (from `growth_above("earnings_growth", x)`), `near_52w_high`.
- Honest framing in the UI: "Research tool, not buy signals."

---

### Task 1: `screen_runner.py` (testable logic)

**Files:**
- Create: `hub/ui/__init__.py`, `hub/ui/screen_runner.py`
- Test: `tests/hub/test_screen_runner.py`

**Interfaces:**
- `PRESET_NAMES: list[str]`.
- `build_criteria(preset: str, overrides: dict) -> list` — starts from `get_preset(preset)`, rebuilds the `pe_below`/`earnings_growth_above`/`near_52w_high` criteria from overrides (`pe_max`, `eps_growth_min`, `near_high_pct`) when present, leaves the rest.
- `screen_to_table(result: dict) -> pandas.DataFrame` — columns `Symbol, Score, Criteria, P/E, EPS growth, Net margin` (None-safe).

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_screen_runner.py
from hub.ui.screen_runner import build_criteria, screen_to_table, PRESET_NAMES

def test_preset_names():
    assert "growth_momentum" in PRESET_NAMES and "value" in PRESET_NAMES

def test_build_criteria_overrides_pe():
    crits = build_criteria("growth_momentum", {"pe_max": 18})
    pe = next(c for c in crits if c.name == "pe_below")
    assert pe.evaluate(None, {"pe_ratio": 15.0}).passed       # 15 < 18 -> pass
    assert not pe.evaluate(None, {"pe_ratio": 25.0}).passed    # 25 < 18 -> fail

def test_build_criteria_ignores_irrelevant_override():
    # 'value' has no near_52w_high; overriding it must not error or add it
    crits = build_criteria("value", {"near_high_pct": 0.03})
    assert all(c.name != "near_52w_high" for c in crits)

def test_screen_to_table_shape_and_none_safe():
    result = {"candidates": [
        {"symbol": "AAA", "composite": 0.71,
         "criteria": {"pe_below": {"passed": True, "value": 20, "score": 1.0},
                      "near_52w_high": {"passed": False, "value": 100, "score": 0.0}},
         "fundamentals": {"pe_ratio": 20.0, "earnings_growth": 0.1, "net_margin": None}}],
        "skipped": []}
    df = screen_to_table(result)
    assert list(df.columns) == ["Symbol", "Score", "Criteria", "P/E", "EPS growth", "Net margin"]
    assert df.iloc[0]["Symbol"] == "AAA" and df.iloc[0]["Criteria"] == "1/2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_screen_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/ui/__init__.py
```

```python
# hub/ui/screen_runner.py
import pandas as pd
from hub.screen.library import PRESETS, get_preset, pe_below, growth_above
from hub.screen.criteria import near_52w_high

PRESET_NAMES = list(PRESETS)

def build_criteria(preset: str, overrides: dict) -> list:
    overrides = overrides or {}
    out = []
    for c in get_preset(preset):
        if c.name == "pe_below" and "pe_max" in overrides:
            out.append(pe_below(overrides["pe_max"]))
        elif c.name == "earnings_growth_above" and "eps_growth_min" in overrides:
            out.append(growth_above("earnings_growth", overrides["eps_growth_min"]))
        elif c.name == "near_52w_high" and "near_high_pct" in overrides:
            out.append(near_52w_high(overrides["near_high_pct"]))
        else:
            out.append(c)
    return out

def screen_to_table(result: dict) -> pd.DataFrame:
    rows = []
    for c in result.get("candidates", []):
        f = c.get("fundamentals") or {}
        crit = c.get("criteria") or {}
        passed = sum(1 for r in crit.values() if r.get("passed"))
        rows.append({
            "Symbol": c["symbol"],
            "Score": round(float(c["composite"]), 3),
            "Criteria": f"{passed}/{len(crit)}",
            "P/E": f.get("pe_ratio"),
            "EPS growth": f.get("earnings_growth"),
            "Net margin": f.get("net_margin"),
        })
    return pd.DataFrame(rows, columns=["Symbol", "Score", "Criteria", "P/E",
                                       "EPS growth", "Net margin"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_screen_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/ui/__init__.py hub/ui/screen_runner.py tests/hub/test_screen_runner.py
git commit -m "feat(hub): testable UI screen_runner (build_criteria + screen_to_table)"
```

---

### Task 2: Streamlit `app.py` + launcher

**Files:**
- Create: `hub/ui/app.py`, `hub/ui/run.py`, `hub/ui/__main__.py`
- Modify: `hub/requirements.txt` (add `streamlit`)
- (No unit test — verified by launch + screenshot)

**Interfaces:**
- `python -m hub.ui` → runs `streamlit run hub/ui/app.py` on port 8501.
- `app.py` renders the dashboard using `screen_runner` + `run_screen` + `get_default_provider`.

- [ ] **Step 1: Write the app**

```python
# hub/ui/app.py
import streamlit as st
import plotly.graph_objects as go
from hub.config import HubConfig
from hub.universe import load_universe
from hub.data.provider import get_default_provider
from hub.screen.screener import run_screen
from hub.ui.screen_runner import build_criteria, screen_to_table, PRESET_NAMES

st.set_page_config(page_title="Stock Research Screener", layout="wide")
st.title("📈 Stock Research Screener")
st.caption("Configurable technical + fundamental screen — a research tool for finding "
           "candidates to investigate, NOT buy signals or predictions.")

cfg = HubConfig.default()

@st.cache_resource(show_spinner=False)
def _provider():
    return get_default_provider(cfg.cache_dir)

@st.cache_data(show_spinner=False)
def _run(universe_name, preset, pe_max, eps_growth_min, near_high_pct, top_k):
    universe = load_universe(universe_name)
    criteria = build_criteria(preset, {"pe_max": pe_max, "eps_growth_min": eps_growth_min,
                                       "near_high_pct": near_high_pct})
    return run_screen(universe, _provider(), criteria, top_k=top_k)

with st.sidebar:
    st.header("Screen settings")
    universe_name = st.selectbox("Universe", ["sp500_sample"])
    preset = st.selectbox("Preset thesis", PRESET_NAMES)
    pe_max = st.slider("Max P/E", 5, 80, 40)
    eps_growth_min = st.slider("Min earnings growth", -0.20, 0.50, 0.10, 0.01)
    near_high_pct = st.slider("Within % of 52w high", 0.01, 0.30, 0.07, 0.01)
    top_k = st.slider("Top K", 5, 50, 20)
    use_llm = st.toggle("Include LLM 'why' (uses API)", value=False)
    run = st.button("Run screen", type="primary", use_container_width=True)

if run or "result" not in st.session_state:
    with st.spinner("Screening — fetching prices + fundamentals…"):
        result = _run(universe_name, preset, pe_max, eps_growth_min, near_high_pct, top_k)
        if use_llm and result["candidates"]:
            import anthropic
            from hub.explain import explain_top
            result = explain_top(result, _provider(), anthropic.Anthropic(), cfg)
        st.session_state["result"] = result

result = st.session_state["result"]
cands = result["candidates"]
st.subheader(f"{len(cands)} matches  ·  {len(result['skipped'])} skipped")

if not cands:
    st.info("No matches — loosen the filters (raise Max P/E, lower Min earnings growth, "
            "or widen the 52-week-high band).")
else:
    st.dataframe(screen_to_table(result), use_container_width=True, hide_index=True)
    sel = st.selectbox("Inspect a candidate", [c["symbol"] for c in cands])
    cand = next(c for c in cands if c["symbol"] == sel)
    left, right = st.columns([2, 1])
    with left:
        try:
            price = _provider().get_ohlcv(sel, 300)
            fig = go.Figure(go.Candlestick(
                x=price.index, open=price["open"], high=price["high"],
                low=price["low"], close=price["close"]))
            fig.update_layout(height=420, xaxis_rangeslider_visible=False,
                              margin=dict(l=0, r=0, t=30, b=0), title=f"{sel} — ~1 year")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart unavailable for {sel}: {e}")
    with right:
        st.markdown(f"**{sel} · score {cand['composite']:.2f}**")
        funds = {k: v for k, v in (cand.get("fundamentals") or {}).items() if v is not None}
        if funds:
            st.markdown("**Fundamentals**")
            st.json(funds, expanded=False)
        st.markdown("**Criteria**")
        for n, r in (cand.get("criteria") or {}).items():
            mark = "✅" if r.get("passed") else "❌"
            st.write(f"{mark} {n} — {r.get('value'):.2f}")
        expl = cand.get("explanation")
        if isinstance(expl, dict) and expl.get("note"):
            st.markdown("**Why (LLM)**")
            st.write(expl.get("note"))
            if expl.get("risk_flags"):
                st.caption("Risks: " + ", ".join(expl["risk_flags"]))
```

```python
# hub/ui/run.py
import os
import sys
import subprocess

def main():
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", app,
                            "--server.headless", "true"])

if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# hub/ui/__main__.py
from hub.ui.run import main
raise SystemExit(main())
```

Add `streamlit` to `hub/requirements.txt`.

- [ ] **Step 2: Verify the app imports cleanly (no run yet)**

Run: `.venv/bin/python -c "import hub.ui.app"`
Expected: no error (Streamlit allows importing the script module; widget calls are no-ops at import outside a run context, or raise a benign warning — if import raises, fix the import-time code).

Note: do NOT block on a long server run here. If `import hub.ui.app` triggers Streamlit's "missing ScriptRunContext" warnings, that's fine — it's not an error. If it raises an ImportError/SyntaxError, fix it.

- [ ] **Step 3: Confirm the full suite still passes**

Run: `.venv/bin/python -m pytest tests/hub/ -q`
Expected: PASS (all hub tests; the app adds no unit tests but must not break collection).

- [ ] **Step 4: Commit**

```bash
git add hub/ui/app.py hub/ui/run.py hub/ui/__main__.py hub/requirements.txt
git commit -m "feat(hub): Streamlit screener dashboard + launcher"
```

---

## Self-Review

**Spec coverage:** screen_runner build_criteria + screen_to_table + PRESET_NAMES (§5 → Task 1) ✓; Streamlit app with sidebar controls, ranked table, per-symbol detail (chart + fundamentals + criteria + LLM) (§4 → Task 2) ✓; launcher `python -m hub.ui` (§3 → Task 2) ✓; empty-result + per-symbol-failure handling (§6 → Task 2) ✓; offline tests for the logic, launch/screenshot for the app (§7 → Tasks 1,2) ✓; streamlit dep (§8 → Task 2) ✓. Deferred per §10: custom criteria builder, saved screens, edgartools, Kronos overlay — not in this step (correct).

**Placeholder scan:** No TBD/TODO; the app + helpers are complete code.

**Type consistency:** `build_criteria -> [Criterion]` (Task 1) consumed by `run_screen` in app.py (Task 2). `screen_to_table -> DataFrame` (Task 1) rendered by `st.dataframe` (Task 2). `PRESET_NAMES` (Task 1) feeds the preset selectbox (Task 2). The app consumes `run_screen`'s candidate shape (`symbol/composite/criteria/fundamentals/explanation`) exactly as produced in Step 2. `get_default_provider`/`load_universe`/`explain_top` reused unchanged. ✓
