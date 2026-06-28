import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_universe(name: str) -> list[str]:
    """Load a ticker list from alpha/data/<name>.txt (one per line, # comments)."""
    path = os.path.join(_DATA_DIR, f"{name}.txt")
    if not os.path.exists(path):
        raise ValueError(f"Unknown universe: {name}")
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]
