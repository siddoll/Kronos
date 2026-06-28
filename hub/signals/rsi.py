from .base import clamp01

class RSI:
    name = "rsi"
    def _rsi(self, c, n=14):
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - 100 / (1 + rs)
    def compute(self, df):
        if len(df) < 15:
            return 0.0
        r = self._rsi(df["close"]).iloc[-1]
        if r != r:  # NaN
            return 0.0
        # peak at ~62, taper to 0 by 85 and below 45
        if r >= 85 or r <= 40:
            return 0.0
        return clamp01(1.0 - abs(r - 62) / 23.0)
