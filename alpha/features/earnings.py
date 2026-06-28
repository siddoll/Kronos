import numpy as np
import pandas as pd


def earnings_drift(earn_df, as_of_dates, decay_days: int = 60) -> dict:
    if earn_df is None or len(earn_df) == 0:
        return {pd.Timestamp(d): np.nan for d in as_of_dates}
    e = earn_df.dropna(subset=["surprise"]).copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values("date")
    out = {}
    for d in as_of_dates:
        d = pd.Timestamp(d)
        prior = e[e["date"] <= d]
        if len(prior) == 0:
            out[d] = np.nan
        else:
            last = prior.iloc[-1]
            days = max(0, (d - pd.Timestamp(last["date"])).days)
            out[d] = float(last["surprise"] * np.exp(-days / decay_days))
    return out
