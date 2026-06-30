from dataclasses import dataclass
from typing import Callable

@dataclass
class CritResult:
    passed: bool
    score: float
    value: float

def _clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x != x:
        return 0.0
    return max(0.0, min(1.0, x))

@dataclass
class Criterion:
    name: str
    kind: str            # "technical" | "fundamental"
    hard: bool
    fn: Callable         # (price_df, fundamentals) -> CritResult
    def evaluate(self, price_df, fundamentals) -> CritResult:
        try:
            return self.fn(price_df, fundamentals)
        except Exception:
            return CritResult(False, 0.0, float("nan"))

_FAIL = lambda: CritResult(False, 0.0, float("nan"))

def near_52w_high(within_pct=0.05, window=252, hard=True) -> Criterion:
    def fn(price, fund):
        c = price["close"]
        if len(c) < window:   # need ~a full year of bars for a real 52-week high
            return _FAIL()
        hi = float(c.tail(window).max()); last = float(c.iloc[-1])
        dist = (hi - last) / hi if hi > 0 else 1.0
        score = _clamp01(1 - dist / within_pct) if within_pct > 0 else 0.0
        return CritResult(dist <= within_pct, score, last)
    return Criterion("near_52w_high", "technical", hard, fn)

def momentum_12_1_positive(hard=True) -> Criterion:
    def fn(price, fund):
        c = price["close"]
        if len(c) < 252:
            return _FAIL()
        r = float(c.iloc[-21] / c.iloc[-252] - 1.0)   # 12 months ago -> 1 month ago
        return CritResult(r > 0, _clamp01(r / 0.5), r)
    return Criterion("momentum_12_1", "technical", hard, fn)

def above_sma(n=200, hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        c = price["close"]
        if len(c) < n:
            return _FAIL()
        s = float(pta.sma(c, length=n).iloc[-1]); last = float(c.iloc[-1])
        ratio = last / s - 1.0 if s > 0 else -1.0
        return CritResult(last > s, _clamp01(ratio / 0.2), last)
    return Criterion(f"above_sma{n}", "technical", hard, fn)

def rsi_between(lo=50, hi=70, hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        c = price["close"]
        if len(c) < 15:
            return _FAIL()
        r = float(pta.rsi(c, length=14).iloc[-1])
        if r != r:
            return _FAIL()
        mid = (lo + hi) / 2
        score = _clamp01(1 - abs(r - mid) / ((hi - lo) / 2)) if hi > lo else 0.0
        return CritResult(lo <= r <= hi, score, r)
    return Criterion("rsi_zone", "technical", hard, fn)

def macd_bullish(hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        c = price["close"]
        if len(c) < 40:
            return _FAIL()
        m = pta.macd(c)
        if m is None or len(m) == 0:
            return _FAIL()
        hcol = [x for x in m.columns if x.startswith("MACDh")][0]
        h = float(m[hcol].iloc[-1])
        if h != h:
            return _FAIL()
        return CritResult(h > 0, 1.0 if h > 0 else 0.0, h)
    return Criterion("macd_bullish", "technical", hard, fn)

def adx_above(n=25, hard=True) -> Criterion:
    import pandas_ta as pta
    def fn(price, fund):
        if len(price) < 30:
            return _FAIL()
        a = pta.adx(price["high"], price["low"], price["close"])
        acol = [x for x in a.columns if x.startswith("ADX_")][0]
        v = float(a[acol].iloc[-1])
        if v != v:
            return _FAIL()
        return CritResult(v >= n, _clamp01(v / 50.0), v)
    return Criterion(f"adx_above{n}", "technical", hard, fn)

def rvol_above(mult=1.5, window=20, hard=False) -> Criterion:
    def fn(price, fund):
        v = price["volume"]
        if len(v) < window + 1:
            return _FAIL()
        avg = float(v.iloc[-(window + 1):-1].mean())
        if avg <= 0:
            return _FAIL()
        ratio = float(v.iloc[-1]) / avg
        return CritResult(ratio >= mult, _clamp01((ratio - 1.0) / 2.0), ratio)
    return Criterion("rvol_above", "technical", hard, fn)

def short_momentum_positive(window=20, hard=True) -> Criterion:
    def fn(price, fund):
        c = price["close"]
        if len(c) < window + 1:
            return _FAIL()
        r = float(c.iloc[-1] / c.iloc[-(window + 1)] - 1.0)
        return CritResult(r > 0, _clamp01(r / 0.20), r)
    return Criterion("short_momentum", "technical", hard, fn)
