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
