from .base import clamp01, atr

class VCP:
    name = "vcp"
    def compute(self, df):
        if len(df) < 50:
            return 0.0
        short = atr(df, 10).iloc[-1]
        long_ = atr(df, 40).iloc[-1]
        if not long_ or long_ <= 0:
            return 0.0
        return clamp01(1.0 - short / long_)  # contraction -> high
