import json
import copy

_SCHEMA = {"type":"object","additionalProperties":False,
    "required":["note","catalyst","bull","bear","risk_flags"],
    "properties":{"note":{"type":"string"},"catalyst":{"type":"string"},
        "bull":{"type":"string"},"bear":{"type":"string"},
        "risk_flags":{"type":"array","items":{"type":"string"}}}}

def explain_candidate(symbol, provider, client, model) -> dict:
    try:
        news = provider.get_news(symbol, 5)
        fund = provider.get_fundamentals(symbol)
        headlines = "\n".join(f"- {n['date']}: {n['title']}" for n in news) or "(none)"
        prompt = (f"Stock {symbol}. Recent headlines:\n{headlines}\n"
                  f"Fundamentals: {fund}\n"
                  "In 1-2 sentences each, give the likely near-term catalyst, a bull case, "
                  "a bear case, and risk_flags (e.g. 'earnings imminent', 'low float', "
                  "'recent dilution', 'possible pump-and-dump'). Be skeptical and concise. "
                  "'note' is a one-line summary.")
        resp = client.messages.create(
            model=model, max_tokens=600,
            messages=[{"role":"user","content":prompt}],
            output_config={"format":{"type":"json_schema","schema":_SCHEMA}})
        text = next(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return json.loads(text)
    except Exception:
        return {"note": "(explanation unavailable)", "catalyst": "", "bull": "",
                "bear": "", "risk_flags": []}

def explain_top(result, provider, client, cfg) -> dict:
    out = copy.deepcopy(result)
    for c in out["candidates"]:
        c["explanation"] = explain_candidate(c["symbol"], provider, client, cfg.explain_model)
        c["fundamentals"] = provider.get_fundamentals(c["symbol"])
    return out
