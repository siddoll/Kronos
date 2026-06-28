import numpy as np
import pandas as pd
from .universe import load_universe
from hub.signals import SIGNALS
from .sectors import load_sectors
from .features.revisions import revision_momentum
from .features.earnings import earnings_drift

def build_panel(provider, cfg, ext_provider=None) -> pd.DataFrame:
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
    reb_dates = [pd.Timestamp(d) for d in calendar[cfg.warmup::cfg.horizon]]

    ext = {}
    if ext_provider is not None:
        for t in frames:
            try:
                ud = ext_provider.get_upgrades_downgrades(t)
                ea = ext_provider.get_earnings(t)
            except Exception:
                ud, ea = None, None
            ext[t] = (revision_momentum(ud, reb_dates), earnings_drift(ea, reb_dates))

    rows = []
    for d in reb_dates:
        for t, df in frames.items():
            pos = df.index.searchsorted(d, side="right") - 1
            if pos < cfg.warmup or pos + cfg.horizon >= len(df):
                continue
            c0 = df["close"].iloc[pos]
            c1 = df["close"].iloc[pos + cfg.horizon]
            if c0 <= 0:
                continue
            feats = {s.name: float(s.compute(df.iloc[:pos + 1])) for s in SIGNALS}
            row = {"date": d, "ticker": t, "sector": sectors.get(t, "UNKNOWN"),
                   **feats, "fwd_ret": c1 / c0 - 1.0}
            if ext_provider is not None:
                row["rev_mom"] = ext[t][0].get(d, np.nan)
                row["pead"] = ext[t][1].get(d, np.nan)
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["fwd_ret"]).reset_index(drop=True)
