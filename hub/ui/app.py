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
            from hub.data.filings import FilingProvider
            from hub.data.kvcache import KVCache
            fp = FilingProvider(kv=KVCache(cfg.cache_dir + "_filings"))
            result = explain_top(result, _provider(), anthropic.Anthropic(), cfg, filing_provider=fp)
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
            val = r.get("value")
            val_str = f"{val:.2f}" if val is not None and val == val else "N/A"
            st.write(f"{mark} {n} — {val_str}")
        expl = cand.get("explanation")
        if isinstance(expl, dict) and expl.get("note"):
            st.markdown("**Why (LLM)**")
            st.write(expl.get("note"))
            if expl.get("risk_flags"):
                st.caption("Risks: " + ", ".join(expl["risk_flags"]))
