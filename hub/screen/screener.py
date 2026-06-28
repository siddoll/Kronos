import numpy as np

def run_screen(universe, provider, criteria, top_k=25, lookback_days=300) -> dict:
    tech = [c for c in criteria if c.kind == "technical"]
    fund = [c for c in criteria if c.kind == "fundamental"]
    candidates, skipped = [], []
    for sym in universe:
        try:
            price = provider.get_ohlcv(sym, lookback_days)
        except Exception as e:
            skipped.append({"symbol": sym, "reason": str(e)}); continue
        if price is None or len(price) < 60:
            skipped.append({"symbol": sym, "reason": "insufficient history"}); continue
        tres = {c.name: c.evaluate(price, None) for c in tech}
        if any(c.hard and not tres[c.name].passed for c in tech):
            continue  # dropped before any fundamental fetch
        fundamentals = provider.get_fundamentals(sym) if fund else {}
        fres = {c.name: c.evaluate(price, fundamentals) for c in fund}
        if any(c.hard and not fres[c.name].passed for c in fund):
            continue
        allres = {**tres, **fres}
        score = float(np.mean([r.score for r in allres.values()])) if allres else 0.0
        candidates.append({
            "symbol": sym, "composite": score,
            "criteria": {n: {"passed": bool(r.passed), "value": r.value, "score": r.score}
                         for n, r in allres.items()},
            "subscores": {n: r.score for n, r in allres.items()},
            "fundamentals": fundamentals, "explanation": None})
    candidates.sort(key=lambda c: c["composite"], reverse=True)
    return {"candidates": candidates[:top_k], "skipped": skipped}
