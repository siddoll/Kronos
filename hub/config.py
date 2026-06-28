from dataclasses import dataclass, field

SIGNAL_NAMES = ("rvol", "breakout", "trend", "vcp", "rsi", "range_exp", "rel_strength")

@dataclass(frozen=True)
class HubConfig:
    universe: str = "sp500_sample"
    lookback_days: int = 260
    top_k: int = 25
    weights: dict = field(default_factory=lambda: {n: 1.0 for n in SIGNAL_NAMES})
    explain_model: str = "claude-haiku-4-5"
    analysis_model: str = "claude-sonnet-4-6"
    out_dir: str = "out"
    cache_dir: str = ".hub_cache"

    @classmethod
    def default(cls) -> "HubConfig":
        return cls()
