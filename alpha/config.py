from dataclasses import dataclass, field
from hub.config import SIGNAL_NAMES, _DEFAULT_WEIGHTS

@dataclass(frozen=True)
class AlphaConfig:
    universe: str = "alpha_sample"
    history_days: int = 1300
    horizon: int = 21
    warmup: int = 60
    n_quantiles: int = 5
    min_names: int = 20
    extra_features: tuple = ()
    cost_bps: float = 10.0
    n_folds: int = 4
    purge: int = 2
    weights: dict = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    out_dir: str = "out_alpha"
    cache_dir: str = ".alpha_cache"

    @classmethod
    def default(cls) -> "AlphaConfig":
        return cls()
