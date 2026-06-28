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
                        if len(ts) > max_chars:
                            out["sections"][s] = ts[:max_chars]
                        else:
                            out["sections"][s] = ts
                except Exception:
                    pass
        except Exception:
            pass
        if self._kv is not None:
            self._kv.put(f"filing_{symbol}", out)
        return out
