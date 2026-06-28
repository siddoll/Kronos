from .base import Signal
from .rvol import RVOL
from .breakout import Breakout
from .trend import Trend
from .vcp import VCP
from .rsi import RSI
from .range_exp import RangeExpansion
from .rel_strength import RelStrength

SIGNALS: list[Signal] = [
    RVOL(), Breakout(), Trend(), VCP(), RSI(), RangeExpansion(), RelStrength(),
]

def score_names() -> list[str]:
    return [s.name for s in SIGNALS]
