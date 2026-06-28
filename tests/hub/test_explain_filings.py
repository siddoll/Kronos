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
