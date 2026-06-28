from .base import clamp01, ema

class Trend:
    name = "trend"
    def compute(self, df):
        if len(df) < 55:
            return 0.0
        c = df["close"]
        ma20, ma50 = ema(c, 20), ema(c, 50)
        score = 0.0
        if c.iloc[-1] > ma50.iloc[-1]:
            score += 0.4
        if ma20.iloc[-1] > ma50.iloc[-1]:
            score += 0.3
        if ma50.iloc[-1] > ma50.iloc[-6]:  # 50MA rising
            score += 0.3
        return clamp01(score)
