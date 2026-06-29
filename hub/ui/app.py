import os
import streamlit as st
import plotly.graph_objects as go
from hub.config import HubConfig
from hub.universe import load_universe
from hub.data.provider import get_default_provider
from hub.screen.screener import run_screen
from hub.screen.forward_test import forward_test
from hub.ui.screen_runner import build_criteria, screen_to_table, PRESET_NAMES, load_screens, save_screen, forward_test_table

SCREENS_PATH = os.path.join(os.path.expanduser("~"), ".hub_screens.json")

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

@st.cache_data(show_spinner=False)
def _forward(universe_name, preset, pe_max, eps_growth_min, near_high_pct):
    universe = load_universe(universe_name)
    provider = _provider()
    frames = {}
    for s in universe:
        try:
            f = provider.get_ohlcv(s, 400)
            if f is not None and len(f) > 280:
                frames[s] = f
        except Exception:
            pass
    criteria = build_criteria(preset, {"pe_max": pe_max, "eps_growth_min": eps_growth_min,
                                       "near_high_pct": near_high_pct})
    return forward_test(frames, criteria, horizons=(5, 10, 20))

# Seed control state once so a loaded screen can pre-fill the widgets (set before they render).
_DEFAULTS = {"preset": PRESET_NAMES[0], "pe_max": 40, "eps_growth_min": 0.10,
             "near_high_pct": 0.07, "top_k": 20}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("Screen settings")
    saved = load_screens(SCREENS_PATH)
    if saved:
        pick = st.selectbox("Load saved screen", ["—"] + list(saved))
        if st.button("📂 Load") and pick != "—":
            for _k, _v in saved[pick].items():
                if _k in _DEFAULTS:
                    st.session_state[_k] = _v
            st.rerun()
    universe_name = st.selectbox("Universe", ["sp500_sample"])
    preset = st.selectbox("Preset thesis", PRESET_NAMES, key="preset")
    pe_max = st.slider("Max P/E", 5, 80, key="pe_max")
    eps_growth_min = st.slider("Min earnings growth", -0.20, 0.50, step=0.01, key="eps_growth_min")
    near_high_pct = st.slider("Within % of 52w high", 0.01, 0.30, step=0.01, key="near_high_pct")
    top_k = st.slider("Top K", 5, 50, key="top_k")
    use_llm = st.toggle("Include LLM 'why' (uses API)", value=False)
    run = st.button("Run screen", type="primary", use_container_width=True)
    st.divider()
    new_name = st.text_input("Save current screen as")
    if st.button("💾 Save screen") and new_name:
        save_screen(new_name, {k: st.session_state[k] for k in _DEFAULTS}, SCREENS_PATH)
        st.success(f"Saved '{new_name}'")

if run or "result" not in st.session_state:
    with st.spinner("Screening — fetching prices + fundamentals…"):
        result = _run(universe_name, preset, pe_max, eps_growth_min, near_high_pct, top_k)
        if use_llm and result["candidates"]:
            import anthropic
            from hub.explain import explain_top
            from hub.data.filings import FilingProvider
            from hub.data.kvcache import KVCache
            fp = FilingProvider(kv=KVCache(cfg.cache_dir + "_filings", ttl_hours=24 * 30))
            result = explain_top(result, _provider(), anthropic.Anthropic(), cfg, filing_provider=fp)
        st.session_state["result"] = result

result = st.session_state["result"]
cands = result["candidates"]
st.subheader(f"{len(cands)} matches  ·  {len(result['skipped'])} skipped")

if not cands:
    st.info("No matches — loosen the filters (raise Max P/E, lower Min earnings growth, "
            "or widen the 52-week-high band).")
else:
    st.download_button("⬇️ Download watchlist (CSV)",
                       screen_to_table(result).to_csv(index=False),
                       file_name="watchlist.csv", mime="text/csv")
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
            val = r.get("value")
            val_str = f"{val:.2f}" if val is not None and val == val else "N/A"
            st.write(f"{mark} {n} — {val_str}")
        expl = cand.get("explanation")
        if isinstance(expl, dict) and expl.get("note"):
            st.markdown("**Why (LLM)**")
            st.write(expl.get("note"))
            if expl.get("risk_flags"):
                st.caption("Risks: " + ", ".join(expl["risk_flags"]))
    with st.expander("⚖️ Reality check — how reliable is this screen? (honest backtest)"):
        st.caption("These picks show strength NOW — this is **not** a prediction that they "
                   "will rise. Below: how this screen's price filters actually performed "
                   "historically (point-in-time, no lookahead).")
        ft = _forward(universe_name, preset, pe_max, eps_growth_min, near_high_pct)
        if ft["n_dates"] == 0 or not any(v["n"] for v in ft["horizons"].values()):
            st.info("Not enough history to backtest this screen.")
        else:
            st.dataframe(forward_test_table(ft), use_container_width=True, hide_index=True)
            e20 = ft["horizons"].get(20, {})
            hr = e20.get("hit_rate", 0) * 100
            edge = e20.get("edge", 0) * 100
            verdict = ("essentially a coin flip — no reliable edge"
                       if abs(edge) < 0.3 or 47 <= hr <= 53 else
                       f"a small historical edge of {edge:+.2f}% (treat with skepticism)")
            st.markdown(f"**At 4 weeks: picks rose {hr:.0f}% of the time → {verdict}.** "
                        "Use this as a starting point for your own research, not a buy signal.")
