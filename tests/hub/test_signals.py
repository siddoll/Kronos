from hub.signals.rvol import RVOL
from hub.signals.breakout import Breakout
from hub.signals.trend import Trend
from hub.signals.vcp import VCP
from hub.signals.rsi import RSI
from hub.signals.range_exp import RangeExpansion
from hub.signals.rel_strength import RelStrength

ALL = [RVOL(), Breakout(), Trend(), VCP(), RSI(), RangeExpansion(), RelStrength()]

def test_all_signals_return_unit_interval(synth_uptrend_df):
    for s in ALL:
        v = s.compute(synth_uptrend_df)
        assert 0.0 <= v <= 1.0, f"{s.name} out of range: {v}"

def test_rvol_high_on_volume_spike(synth_uptrend_df):
    assert RVOL().compute(synth_uptrend_df) > 0.5  # last bar volume is 5x

def test_trend_high_in_uptrend(synth_uptrend_df):
    assert Trend().compute(synth_uptrend_df) > 0.5

def test_breakout_high_near_high(synth_uptrend_df):
    assert Breakout().compute(synth_uptrend_df) > 0.8  # rising series ends at its high

def test_signals_do_not_mutate_input(synth_uptrend_df):
    before = synth_uptrend_df.copy()
    for s in ALL:
        s.compute(synth_uptrend_df)
    assert synth_uptrend_df.equals(before)
