import numpy as np
from hub.signals.base import clamp01, atr

def test_clamp01_bounds():
    assert clamp01(-1) == 0.0 and clamp01(2) == 1.0 and clamp01(0.5) == 0.5
    assert clamp01(np.nan) == 0.0

def test_atr_positive(make_df):
    df = make_df(list(range(10, 40)))
    a = atr(df, 14)
    assert a.iloc[-1] > 0
