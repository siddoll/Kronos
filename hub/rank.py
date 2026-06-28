def score_ticker(df, signals, weights: dict) -> dict:
    subscores = {s.name: float(s.compute(df)) for s in signals}
    total_w = sum(weights.get(name, 0.0) for name in subscores) or 1.0
    composite = sum(subscores[name] * weights.get(name, 0.0)
                    for name in subscores) / total_w
    return {"subscores": subscores, "composite": composite}

def rank_candidates(results: dict, top_k: int) -> list:
    rows = [{"symbol": sym, "composite": r["composite"],
             "subscores": r["subscores"]} for sym, r in results.items()]
    rows.sort(key=lambda r: r["composite"], reverse=True)
    return rows[:top_k]
