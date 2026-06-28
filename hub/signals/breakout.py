from .base import clamp01

class Breakout:
    name = "breakout"
    def compute(self, df, n: int = 55):
        if len(df) < n:
            return 0.0
        window = df["close"].iloc[-n:]
        lo, hi = window.min(), window.max()
        if hi <= lo:
            return 0.0
        return clamp01((df["close"].iloc[-1] - lo) / (hi - lo))
