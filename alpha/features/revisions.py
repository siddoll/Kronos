import numpy as np
import pandas as pd


def revision_momentum(ud_df, as_of_dates, window_days: int = 90) -> dict:
    if ud_df is None or len(ud_df) == 0:
        return {pd.Timestamp(d): np.nan for d in as_of_dates}
    dt = pd.to_datetime(ud_df["date"])
    out = {}
    for d in as_of_dates:
        d = pd.Timestamp(d)
        lo = d - pd.Timedelta(days=window_days)
        m = (dt > lo) & (dt <= d)
        out[d] = float(ud_df.loc[m, "up"].sum() - ud_df.loc[m, "down"].sum()) if m.any() else np.nan
    return out
