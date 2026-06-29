import os

# (output_key, edgartools TenK attribute) — MD&A's real attribute is management_discussion.
_SECTIONS = (("risk_factors", "risk_factors"),
             ("business", "business"),
             ("mda", "management_discussion"))
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
            for key, attr in _SECTIONS:
                try:
                    t = getattr(obj, attr, None)
                    if t:
                        ts = " ".join(str(t).split())
                        if ts:  # skip whitespace-only sections
                            out["sections"][key] = ts[:max_chars]
                except Exception:
                    pass
        except Exception:
            pass
        if self._kv is not None:
            self._kv.put(f"filing_{symbol}", out)
        return out
