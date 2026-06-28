import numpy as np
from alpha.neutralize import neutralize

def test_neutralized_means_are_zero(planted_panel):
    resid = neutralize(planted_panel)
    df = planted_panel.assign(_r=resid.values)
    # per-date mean ~0
    assert df.groupby("date")["_r"].mean().abs().max() < 1e-9
    # per-(date,sector) mean ~0
    assert df.groupby(["date","sector"])["_r"].mean().abs().max() < 1e-9
