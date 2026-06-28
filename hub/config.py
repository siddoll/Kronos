from dataclasses import dataclass, field

SIGNAL_NAMES = ("rvol", "breakout", "trend", "vcp", "rsi", "range_exp", "rel_strength")

# Spec §4: tilt the composite toward the classic early-mover tells.
_DEFAULT_WEIGHTS = {
    "rvol": 1.5, "breakout": 1.5, "rel_strength": 1.5,
    "trend": 1.0, "vcp": 1.0, "rsi": 1.0, "range_exp": 1.0,
}

@dataclass(frozen=True)
class HubConfig:
    universe: str = "sp500_sample"
    lookback_days: int = 260
    top_k: int = 25
    weights: dict = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    explain_model: str = "claude-haiku-4-5"
    analysis_model: str = "claude-sonnet-4-6"
    out_dir: str = "out"
    cache_dir: str = ".hub_cache"

    @classmethod
    def default(cls) -> "HubConfig":
        return cls()
