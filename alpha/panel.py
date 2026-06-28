import numpy as np
import pandas as pd
from hub.universe import load_universe
from hub.signals import SIGNALS
from .sectors import load_sectors

def build_panel(provider, cfg) -> pd.DataFrame:
    sectors = load_sectors()
    frames = {}
    for t in load_universe(cfg.universe):
        try:
            df = provider.get_ohlcv(t, cfg.history_days)
            if df is not None and len(df) > cfg.warmup + cfg.horizon:
                frames[t] = df
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    calendar = sorted(set().union(*[set(df.index) for df in frames.values()]))
    reb_dates = calendar[cfg.warmup::cfg.horizon]
    rows = []
    for d in reb_dates:
        for t, df in frames.items():
            pos = df.index.searchsorted(d, side="right") - 1
            if pos < cfg.warmup or pos + cfg.horizon >= len(df):
                continue
            window = df.iloc[:pos + 1]
            c0 = df["close"].iloc[pos]
            c1 = df["close"].iloc[pos + cfg.horizon]
            if c0 <= 0:
                continue
            feats = {s.name: float(s.compute(window)) for s in SIGNALS}
            rows.append({"date": pd.Timestamp(d), "ticker": t,
                         "sector": sectors.get(t, "UNKNOWN"),
                         **feats, "fwd_ret": c1 / c0 - 1.0})
    return pd.DataFrame(rows).dropna(subset=["fwd_ret"]).reset_index(drop=True)
