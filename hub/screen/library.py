from .criteria import Criterion, CritResult, _clamp01, near_52w_high, \
    momentum_12_1_positive, above_sma, rsi_between, rvol_above, short_momentum_positive

def _fund(name, key, op, threshold, hard=True, scale=None) -> Criterion:
    def fn(price, fund):
        v = (fund or {}).get(key)
        if v is None:
            return CritResult(False, 0.0, float("nan"))   # fail closed
        try:
            v = float(v)
        except (TypeError, ValueError):
            return CritResult(False, 0.0, float("nan"))
        passed = v < threshold if op == "<" else v > threshold
        if scale:
            delta = (threshold - v) if op == "<" else (v - threshold)
            score = _clamp01(delta / abs(scale) + 0.5)
        else:
            score = 1.0 if passed else 0.0
        return CritResult(passed, score, v)
    return Criterion(name, "fundamental", hard, fn)

def pe_below(x=40, hard=True):
    return _fund("pe_below", "pe_ratio", "<", x, hard, scale=x)

def peg_below(x=2, hard=True):
    return _fund("peg_below", "peg_ratio", "<", x, hard, scale=x)

def growth_above(key="earnings_growth", x=0.1, hard=True):
    return _fund(f"{key}_above", key, ">", x, hard, scale=0.3)

def margin_above(key="net_margin", x=0.1, hard=True):
    return _fund(f"{key}_above", key, ">", x, hard, scale=0.3)

def mktcap_above(x=2e9, hard=True):
    return _fund("mktcap_above", "market_cap", ">", x, hard, scale=x)

PRESETS = {
    "growth_momentum": [near_52w_high(0.07), momentum_12_1_positive(), above_sma(200),
                        growth_above("earnings_growth", 0.10), pe_below(40)],
    "value": [pe_below(15), margin_above("net_margin", 0.10),
              growth_above("revenue_growth", 0.0), mktcap_above(2e9)],
    "quality_momentum": [above_sma(200), rsi_between(50, 70),
                         margin_above("net_margin", 0.15), growth_above("earnings_growth", 0.0)],
    "momentum_catalyst": [rvol_above(1.5, hard=False), near_52w_high(0.12),
                          above_sma(50), short_momentum_positive(20)],
}

def get_preset(name: str):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return PRESETS[name]
