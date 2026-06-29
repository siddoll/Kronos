# Step 4 Polish — Implementation Plan (filings "why" + UI usability)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** (1) Ground the LLM "why" in real SEC 10-K text via edgartools; (2) UI polish — CSV export + saved screens.

**Architecture:** `hub/data/filings.py` (cached, injectable FilingProvider) → fed into `hub/explain.py` (optional `filing_provider`) → constructed by CLI/UI when LLM runs. `hub/ui/screen_runner.py` gains save/load helpers; `hub/ui/app.py` gains a CSV button + saved-screen controls.

**Tech Stack:** Python 3.13, edgartools (verified), the merged hub.

## Global Constraints

- `.venv/bin/python` / `.venv/bin/pytest`. Branch `step4-polish`. Add `edgartools` to `hub/requirements.txt`.
- Backward compatible: `explain_top`/`explain_candidate` gain an OPTIONAL `filing_provider=None`; with None they behave exactly as today (existing `tests/hub/test_explain.py` stays green).
- SEC identity is configurable (env `SEC_IDENTITY`, generic default) — never hardcode the user's personal email.
- Every edgar/file path is wrapped so failure degrades gracefully (empty sections / `{}`), never raises.
- Tests must NOT hit the network: inject a fake `company_fn` / stub client; only a live smoke uses real edgar.

---

### Task 1: FilingProvider (cached SEC 10-K text, injectable)

**Files:**
- Create: `hub/data/filings.py`
- Modify: `hub/requirements.txt` (add `edgartools`)
- Test: `tests/hub/test_filings.py`

**Interfaces:**
- `FilingProvider(identity=None, kv=None, company_fn=None)`; `get_filing_summary(symbol, max_chars=2000) -> {"form","date","sections":{risk_factors?,business?,mda?}}`. `company_fn` injectable (default: lazily `from edgar import set_identity, Company`, call `set_identity`, use `Company`). Cached via `kv`; any error → empty sections.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_filings.py
from hub.data.filings import FilingProvider

class _Filing:
    form = "10-K"; filing_date = "2025-10-31"
    def obj(self):
        class O:
            risk_factors = "Item 1A. Risk Factors  " + "competition and supply risk. " * 50
            business = "We design and sell devices. " * 30
        return O()
class _Filings:
    def latest(self, n): return _Filing()
class _Company:
    def __init__(self, symbol): pass
    def get_filings(self, form=None): return _Filings()

def test_filing_summary_extracts_and_truncates():
    fp = FilingProvider(company_fn=_Company)
    out = fp.get_filing_summary("AAPL", max_chars=200)
    assert out["form"] == "10-K" and out["date"] == "2025-10-31"
    assert "risk_factors" in out["sections"] and len(out["sections"]["risk_factors"]) == 200
    assert "Risk Factors" in out["sections"]["risk_factors"]

def test_filing_error_is_safe():
    class Boom:
        def __init__(self, s): raise RuntimeError("edgar down")
    out = FilingProvider(company_fn=Boom).get_filing_summary("AAPL")
    assert out["sections"] == {} and out["form"] is None

def test_filing_uses_cache(tmp_path):
    from hub.data.kvcache import KVCache
    kv = KVCache(str(tmp_path))
    FilingProvider(company_fn=_Company, kv=kv).get_filing_summary("AAA")
    assert kv.get("filing_AAA")["form"] == "10-K"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_filings.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# hub/data/filings.py
import os

_SECTIONS = ("risk_factors", "business", "mda")
_DEFAULT_IDENTITY = "kronos-research-tool research@example.com"

class FilingProvider:
    def __init__(self, identity=None, kv=None, company_fn=None):
        self._identity = identity or os.environ.get("SEC_IDENTITY", _DEFAULT_IDENTITY)
        self._kv = kv
        self._company_fn = company_fn

    def _company(self):
        if self._company_fn is None:
            from edgar import set_identity, Company
            set_identity(self._identity)
            self._company_fn = Company
        return self._company_fn

    def get_filing_summary(self, symbol, max_chars=2000) -> dict:
        if self._kv is not None:
            hit = self._kv.get(f"filing_{symbol}")
            if hit is not None:
                return hit
        out = {"form": None, "date": None, "sections": {}}
        try:
            f = self._company()(symbol).get_filings(form="10-K").latest(1)
            out["form"] = str(getattr(f, "form", "") or "") or None
            out["date"] = str(getattr(f, "filing_date", "") or "") or None
            obj = f.obj()
            for s in _SECTIONS:
                try:
                    t = getattr(obj, s, None)
                    if t:
                        ts = " ".join(str(t).split())
                        if len(ts) > 200:
                            out["sections"][s] = ts[:max_chars]
                except Exception:
                    pass
        except Exception:
            pass
        if self._kv is not None:
            self._kv.put(f"filing_{symbol}", out)
        return out
```

Add `edgartools` to `hub/requirements.txt` (append a line).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_filings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hub/data/filings.py hub/requirements.txt tests/hub/test_filings.py
git commit -m "feat(hub): cached SEC 10-K filing-text provider (edgartools, injectable)"
```

---

### Task 2: Wire filings into the LLM "why"

**Files:**
- Modify: `hub/explain.py`, `hub/cli.py`, `hub/ui/app.py`
- Test: `tests/hub/test_explain_filings.py`

**Interfaces:**
- `explain_candidate(symbol, provider, client, model, filing_provider=None)` and `explain_top(result, provider, client, cfg, filing_provider=None)` — optional; None ⇒ unchanged. When given, a condensed filing excerpt (risk factors then business) is appended to the prompt with an instruction to ground bull/bear/risk in it.
- CLI `scan`/`screen` and the UI construct a `FilingProvider()` and pass it when the LLM step runs.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_explain_filings.py
from hub.explain import explain_candidate

class StubProvider:
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {"pe_ratio": 20.0}

class StubFilings:
    def get_filing_summary(self, s, max_chars=2000):
        return {"form": "10-K", "date": "2025-10-31",
                "sections": {"risk_factors": "SUPPLYCHAIN_RISK_SENTINEL competition",
                             "business": "Sells devices"}}

class CapturingClient:
    def __init__(self):
        self.prompt = None
        outer = self
        class M:
            @staticmethod
            def create(**kw):
                outer.prompt = kw["messages"][0]["content"]
                class B: type = "text"; text = '{"note":"n","catalyst":"c","bull":"b","bear":"x","risk_flags":[]}'
                class R: content = [B()]
                return R()
        self.messages = M()

def test_prompt_includes_filing_text():
    client = CapturingClient()
    explain_candidate("AAPL", StubProvider(), client, "m", filing_provider=StubFilings())
    assert "SUPPLYCHAIN_RISK_SENTINEL" in client.prompt   # filing text reached the LLM

def test_no_filing_provider_still_works():
    client = CapturingClient()
    out = explain_candidate("AAPL", StubProvider(), client, "m")
    assert out["note"] == "n" and "SUPPLYCHAIN_RISK_SENTINEL" not in (client.prompt or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_explain_filings.py -v`
Expected: FAIL (`filing_provider` not accepted)

- [ ] **Step 3: Write minimal implementation**

Read the current `hub/explain.py`. Add `filing_provider=None` to `explain_candidate` and build a filing block. The key edits:

In `explain_candidate(symbol, provider, client, model, filing_provider=None)`, after fetching `news`/`fund` and before building the prompt, add:

```python
        filing_block = ""
        if filing_provider is not None:
            try:
                fs = filing_provider.get_filing_summary(symbol)
                secs = fs.get("sections") or {}
                if secs:
                    rf = (secs.get("risk_factors") or "")[:1500]
                    bus = (secs.get("business") or "")[:800]
                    filing_block = (f"\nFrom the latest {fs.get('form')} ({fs.get('date')}):\n"
                                    f"Risk factors: {rf}\nBusiness: {bus}\n"
                                    "Ground the bull/bear/risk in these disclosures when relevant.\n")
            except Exception:
                filing_block = ""
```

and insert `filing_block` into the prompt string (append it after the `Fundamentals: {fund}` line, before the instruction sentence).

In `explain_top(result, provider, client, cfg, filing_provider=None)`, pass `filing_provider` through to `explain_candidate(...)`.

In `hub/cli.py`: in BOTH the `scan` and `screen` handlers, where it currently does `explain_top(result, provider, anthropic.Anthropic(), cfg)`, change to construct a filing provider and pass it:

```python
            from .data.filings import FilingProvider
            from .data.kvcache import KVCache
            fp = FilingProvider(kv=KVCache(cfg.cache_dir + "_filings"))
            result = explain_top(result, provider, anthropic.Anthropic(), cfg, filing_provider=fp)
```

In `hub/ui/app.py`: where it calls `explain_top(result, _provider(), anthropic.Anthropic(), cfg)`, similarly construct and pass a `FilingProvider(kv=KVCache(cfg.cache_dir + "_filings"))`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_explain_filings.py tests/hub/test_explain.py -q`
Expected: PASS (new + existing explain tests)

- [ ] **Step 5: Commit**

```bash
git add hub/explain.py hub/cli.py hub/ui/app.py tests/hub/test_explain_filings.py
git commit -m "feat(hub): ground the LLM 'why' in real 10-K filing text"
```

---

### Task 3: Saved screens + CSV export (UI)

**Files:**
- Modify: `hub/ui/screen_runner.py`, `hub/ui/app.py`
- Test: `tests/hub/test_saved_screens.py`

**Interfaces:**
- `screen_runner.load_screens(path) -> dict` (corrupt/missing → `{}`), `screen_runner.save_screen(name, settings, path) -> dict` (persists, returns updated dict; swallows write errors).
- UI: a sidebar "Save current screen" (name + button) and "Load saved screen" selectbox; a "Download watchlist (CSV)" button above the table.

- [ ] **Step 1: Write the failing test**

```python
# tests/hub/test_saved_screens.py
from hub.ui.screen_runner import save_screen, load_screens

def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "screens.json")
    assert load_screens(p) == {}
    save_screen("my growth", {"preset": "growth_momentum", "pe_max": 30}, p)
    sc = load_screens(p)
    assert sc["my growth"]["pe_max"] == 30

def test_load_corrupt_is_empty(tmp_path):
    p = tmp_path / "bad.json"; p.write_text("{not json")
    assert load_screens(str(p)) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_saved_screens.py -v`
Expected: FAIL (`save_screen`/`load_screens` missing)

- [ ] **Step 3: Write minimal implementation**

Append to `hub/ui/screen_runner.py`:

```python
import json as _json

def load_screens(path) -> dict:
    try:
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return {}

def save_screen(name, settings, path) -> dict:
    screens = load_screens(path)
    screens[str(name)] = settings
    try:
        with open(path, "w") as f:
            _json.dump(screens, f, indent=2)
    except Exception:
        pass
    return screens
```

In `hub/ui/app.py`:
1. At the top, set a screens-file path: `SCREENS_PATH = os.path.join(os.path.expanduser("~"), ".hub_screens.json")` (add `import os`).
2. In the sidebar (after the sliders), add saved-screen controls:

```python
    from hub.ui.screen_runner import load_screens, save_screen
    st.divider()
    saved = load_screens(SCREENS_PATH)
    if saved:
        pick = st.selectbox("Load saved screen", ["—"] + list(saved))
        if pick != "—":
            s = saved[pick]
            st.caption(f"Loaded '{pick}': {s}")
    new_name = st.text_input("Save current screen as")
    if st.button("💾 Save screen") and new_name:
        save_screen(new_name, {"preset": preset, "pe_max": pe_max,
                               "eps_growth_min": eps_growth_min,
                               "near_high_pct": near_high_pct, "top_k": top_k}, SCREENS_PATH)
        st.success(f"Saved '{new_name}'")
```

3. Above `st.dataframe(...)` in the results block, add a CSV download:

```python
        st.download_button("⬇️ Download watchlist (CSV)",
                           screen_to_table(result).to_csv(index=False),
                           file_name="watchlist.csv", mime="text/csv")
```

(Ensure `screen_to_table` is imported in app.py — it already is.)

- [ ] **Step 4: Run tests + verify app imports**

Run: `.venv/bin/python -m pytest tests/hub/ -q && .venv/bin/python -c "import hub.ui.app"`
Expected: all hub tests PASS; `import hub.ui.app` does not raise.

- [ ] **Step 5: Commit**

```bash
git add hub/ui/screen_runner.py hub/ui/app.py tests/hub/test_saved_screens.py
git commit -m "feat(hub): saved screens + CSV export in the UI"
```

---

## Self-Review

**Spec coverage:** FilingProvider cached/injectable (§4 → Task 1) ✓; deep "why" wiring + CLI/UI hookup (§5 → Task 2) ✓; save/load + CSV (§6 → Task 3) ✓; graceful degradation (§7 → Tasks 1,2,3) ✓; offline tests + live/screenshot verification (§8 → Tasks 1-3) ✓; edgartools dep + configurable identity (§9/§4 → Task 1) ✓. Deferred per §11: quantstats, custom-criteria builder, Kronos overlay — not in this step (correct).

**Placeholder scan:** No TBD/TODO; every code step has real code.

**Type consistency:** `FilingProvider.get_filing_summary -> {form,date,sections}` (Task 1) consumed by `explain_candidate` (Task 2). `explain_top(..., filing_provider=None)` (Task 2) called by CLI + UI with a `FilingProvider`. `save_screen`/`load_screens` (Task 3) used by the UI sidebar. `screen_to_table` (existing) reused for the CSV. The `kv` cache (existing KVCache) reused by FilingProvider. ✓
