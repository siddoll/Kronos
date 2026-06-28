from hub.explain import explain_top

class StubProvider:
    def get_news(self, s, limit=5): return [{"date":"2026-06-27","title":"X beats earnings","source":"PR"}]
    def get_fundamentals(self, s): return {"market_cap": 1e9}

class StubClient:
    class messages:
        @staticmethod
        def create(**kw):
            class B:  # minimal content block with valid JSON text
                type="text"; text='{"note":"earnings beat","catalyst":"earnings","bull":"x","bear":"y","risk_flags":["earnings imminent"]}'
            class R: content=[B()]
            return R()

def test_explain_top_sets_explanation():
    from hub.config import HubConfig
    result = {"candidates":[{"symbol":"X","composite":0.8,"subscores":{},"explanation":None}],"skipped":[]}
    out = explain_top(result, StubProvider(), StubClient(), HubConfig.default())
    e = out["candidates"][0]["explanation"]
    assert e["note"] == "earnings beat" and "earnings imminent" in e["risk_flags"]
