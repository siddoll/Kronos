from .base import Signal

SIGNALS: list[Signal] = []

def score_names() -> list[str]:
    return [s.name for s in SIGNALS]
