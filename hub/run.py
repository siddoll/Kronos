from .universe import load_universe
from .signals import SIGNALS
from .rank import score_ticker, rank_candidates

def scan(cfg, provider) -> dict:
    symbols = load_universe(cfg.universe)
    results, skipped = {}, []
    for sym in symbols:
        try:
            df = provider.get_ohlcv(sym, cfg.lookback_days)
            if df is None or len(df) < 55:
                skipped.append({"symbol": sym, "reason": "insufficient history"})
                continue
            results[sym] = score_ticker(df, SIGNALS, cfg.weights)
        except Exception as e:  # isolate per-symbol failures
            skipped.append({"symbol": sym, "reason": str(e)})
    candidates = rank_candidates(results, cfg.top_k)
    for c in candidates:
        c.setdefault("explanation", None)
    return {"candidates": candidates, "skipped": skipped}
