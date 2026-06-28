import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def rank_ic_series(dates, x, y) -> pd.Series:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "x": np.asarray(x, float),
                       "y": np.asarray(y, float)}).dropna()
    out = {}
    for d, g in df.groupby("date"):
        if len(g) < 3:
            continue
        ic = spearmanr(g["x"], g["y"]).correlation
        if ic == ic:  # not NaN
            out[d] = ic
    return pd.Series(out, dtype=float)

def ic_summary(ic_series: pd.Series) -> dict:
    n = int(len(ic_series))
    if n == 0:
        return {"mean_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "n": 0}
    mean = float(ic_series.mean())
    std = float(ic_series.std(ddof=1)) if n > 1 else 0.0
    ir = mean / std if std > 0 else 0.0
    return {"mean_ic": mean, "ic_ir": ir, "t_stat": ir * np.sqrt(n), "n": n}
