import numpy as np
from .signals import SIGNALS
from .rank import score_ticker, rank_candidates

def backtest_screen(frames: dict, cfg, horizon: int = 10, step: int = 5) -> dict:
    lookback = cfg.lookback_days
    topk_rets, univ_rets, hits = [], [], []
    any_symbol = next(iter(frames.values()))
    n_bars = len(any_symbol)
    for origin in range(lookback, n_bars - horizon, step):
        scored, fwd = {}, {}
        for sym, df in frames.items():
            if len(df) < origin + horizon:
                continue
            window = df.iloc[:origin]
            if len(window) < 55:
                continue
            scored[sym] = score_ticker(window, SIGNALS, cfg.weights)
            c0 = df["close"].iloc[origin - 1]
            c1 = df["close"].iloc[origin - 1 + horizon]
            fwd[sym] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        if not scored:
            continue
        top = [r["symbol"] for r in rank_candidates(scored, cfg.top_k)]
        topk_rets.append(np.mean([fwd[s] for s in top]))
        univ_rets.append(np.mean(list(fwd.values())))
        hits.append(np.mean([fwd[s] > 0 for s in top]))
    if not topk_rets:
        return {"n": 0, "topk_fwd_return": 0.0, "universe_fwd_return": 0.0,
                "edge": 0.0, "hit_rate": 0.0}
    tk, uv = float(np.mean(topk_rets)), float(np.mean(univ_rets))
    return {"n": len(topk_rets), "topk_fwd_return": tk, "universe_fwd_return": uv,
            "edge": tk - uv, "hit_rate": float(np.mean(hits))}
