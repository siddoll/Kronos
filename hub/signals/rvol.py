from .base import clamp01

class RVOL:
    name = "rvol"
    def compute(self, df):
        if len(df) < 21:
            return 0.0
        avg = df["volume"].iloc[-21:-1].mean()
        if avg <= 0:
            return 0.0
        ratio = df["volume"].iloc[-1] / avg
        return clamp01((ratio - 1.0) / 2.0)  # 1x->0, 3x->1
