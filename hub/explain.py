import json
import copy

_SCHEMA = {"type":"object","additionalProperties":False,
    "required":["note","catalyst","bull","bear","risk_flags"],
    "properties":{"note":{"type":"string"},"catalyst":{"type":"string"},
        "bull":{"type":"string"},"bear":{"type":"string"},
        "risk_flags":{"type":"array","items":{"type":"string"}}}}

def explain_candidate(symbol, provider, client, model, filing_provider=None) -> dict:
    try:
        news = provider.get_news(symbol, 5)
        fund = provider.get_fundamentals(symbol)
        headlines = "\n".join(f"- {n['date']}: {n['title']}" for n in news) or "(none)"
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
        prompt = (f"Stock {symbol}. Recent headlines:\n{headlines}\n"
                  f"Fundamentals: {fund}\n"
                  f"{filing_block}"
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

def explain_top(result, provider, client, cfg, filing_provider=None) -> dict:
    out = copy.deepcopy(result)
    for c in out["candidates"]:
        c["explanation"] = explain_candidate(c["symbol"], provider, client, cfg.explain_model,
                                             filing_provider=filing_provider)
        c["fundamentals"] = provider.get_fundamentals(c["symbol"])
    return out
