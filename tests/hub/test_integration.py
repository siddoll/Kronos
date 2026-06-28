import os
import numpy as np
from hub.config import HubConfig
from hub.run import scan
from hub.report import write_reports

class StubProvider:
    def __init__(self, frames): self.frames = frames
    def get_ohlcv(self, s, n): return self.frames[s]
    def get_news(self, s, limit=5): return []
    def get_fundamentals(self, s): return {}

def test_end_to_end_scan_to_reports(tmp_path, make_df, monkeypatch):
    import hub.run as run_mod
    syms = [f"S{i}" for i in range(10)]
    monkeypatch.setattr(run_mod, "load_universe", lambda name: syms)
    frames = {s: make_df(list(np.linspace(50 + i, 120 + i, 200)))
              for i, s in enumerate(syms)}
    cfg = HubConfig(out_dir=str(tmp_path), top_k=5)
    result = scan(cfg, StubProvider(frames))
    assert len(result["candidates"]) == 5
    paths = write_reports(result, cfg, "20260628")
    assert os.path.exists(paths["html"]) and os.path.exists(paths["csv"])
