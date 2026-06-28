from .base import clamp01

class RangeExpansion:
    name = "range_exp"
    def compute(self, df):
        if len(df) < 21:
            return 0.0
        rng = (df["high"] - df["low"])
        avg = rng.iloc[-21:-1].mean()
        if avg <= 0:
            return 0.0
        return clamp01((rng.iloc[-1] / avg - 1.0) / 1.5)  # 1x->0, 2.5x->1
