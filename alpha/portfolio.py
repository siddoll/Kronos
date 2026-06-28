import numpy as np
import pandas as pd


def long_short(panel, score, n_quantiles=5, cost_bps=10.0,
               raw_target="fwd_ret", periods_per_year=12) -> dict:
    df = panel[["date", "ticker", raw_target]].copy()
    df["score"] = np.asarray(score, float)
    df = df.dropna(subset=["score"])
    gross, turn, dates = [], [], []
    prev_long, prev_short = set(), set()
    for d, g in df.groupby("date"):
        if len(g) < n_quantiles:
            continue
        g = g.sort_values("score")
        k = max(1, len(g) // n_quantiles)
        short = g.iloc[:k]
        long_ = g.iloc[-k:]
        ret = long_[raw_target].mean() - short[raw_target].mean()
        ls, ss = set(long_["ticker"]), set(short["ticker"])
        denom = len(ls) + len(ss)
        changed = len(ls ^ prev_long) + len(ss ^ prev_short)
        t = changed / denom if denom else 0.0
        prev_long, prev_short = ls, ss
        gross.append(ret)
        turn.append(t)
        dates.append(d)
    gross = pd.Series(gross, index=pd.to_datetime(dates))
    turnover = pd.Series(turn, index=gross.index)
    net = gross - cost_bps / 1e4 * turnover
    sd = net.std(ddof=1)
    spp = net.mean() / sd if sd > 0 else 0.0
    return {"gross": gross, "net": net, "turnover": turnover,
            "sharpe_per_period": spp, "sharpe": spp * np.sqrt(periods_per_year),
            "equity": (1 + net).cumprod()}
