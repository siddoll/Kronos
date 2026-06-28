from .base import clamp01

class RelStrength:
    name = "rel_strength"
    def compute(self, df, n: int = 60):
        if len(df) < n + 1:
            return 0.0
        past = df["close"].iloc[-n-1]
        if past <= 0:
            return 0.0
        ret = df["close"].iloc[-1] / past - 1.0
        return clamp01(ret / 0.25)  # +25% over n bars -> 1.0
