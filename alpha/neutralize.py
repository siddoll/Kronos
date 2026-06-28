import pandas as pd

def neutralize(panel: pd.DataFrame, target: str = "fwd_ret") -> pd.Series:
    g = panel.groupby(["date", "sector"])[target]
    resid = panel[target] - g.transform("mean")
    return resid
