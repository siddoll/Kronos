import numpy as np

def _zero():
    return {"hit_rate": 0.0, "pick_return": 0.0, "market_return": 0.0, "edge": 0.0, "n": 0}

def forward_test(frames, criteria, horizons=(5, 10, 20), step=5, warmup=252) -> dict:
    hard_tech = [c for c in criteria if c.kind == "technical" and c.hard]
    horizons = tuple(horizons)
    if not frames or not hard_tech:
        return {"horizons": {h: _zero() for h in horizons}, "n_dates": 0,
                "n_names": len(frames)}
    n_bars = max(len(df) for df in frames.values())
    max_h = max(horizons)
    pick_rets = {h: [] for h in horizons}
    mkt_rets = {h: [] for h in horizons}
    hits = {h: [] for h in horizons}
    n_dates = 0
    for origin in range(warmup, n_bars - max_h, step):
        fwd = {h: {} for h in horizons}
        picks = []
        any_data = False
        for t, df in frames.items():
            if origin >= len(df):
                continue
            window = df.iloc[:origin]
            if len(window) < warmup:
                continue
            c0 = df["close"].iloc[origin - 1]
            got = {}
            for h in horizons:
                j = origin - 1 + h
                if j < len(df) and c0 > 0:
                    r = float(df["close"].iloc[j] / c0 - 1.0)
                    if r == r:  # skip if NaN
                        got[h] = r
            if not got:
                continue
            any_data = True
            for h, r in got.items():
                fwd[h][t] = r
            if all(c.evaluate(window, None).passed for c in hard_tech):
                picks.append(t)
        if not any_data:
            continue
        n_dates += 1
        for h in horizons:
            allr = list(fwd[h].values())
            if not allr:
                continue
            mkt = float(np.mean(allr))
            pr = [fwd[h][t] for t in picks if t in fwd[h]]
            if pr:
                pick_rets[h].append(float(np.mean(pr)))
                mkt_rets[h].append(mkt)
                hits[h].extend([1.0 if x > 0 else 0.0 for x in pr])
    out = {"horizons": {}, "n_dates": n_dates, "n_names": len(frames)}
    for h in horizons:
        if pick_rets[h]:
            p, m = float(np.mean(pick_rets[h])), float(np.mean(mkt_rets[h]))
            out["horizons"][h] = {"hit_rate": float(np.mean(hits[h])), "pick_return": p,
                                  "market_return": m, "edge": p - m, "n": len(hits[h])}
        else:
            out["horizons"][h] = _zero()
    return out
