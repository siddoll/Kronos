import pandas as pd
from hub.config import HubConfig
from hub.run import scan
from hub.report import write_reports

class StubProvider:
    def __init__(self, frames): self.frames = frames
    def get_ohlcv(self, symbol, lookback_days):
        if symbol not in self.frames:
            raise RuntimeError("no data")
        return self.frames[symbol]
    def get_news(self, symbol, limit=5): return []
    def get_fundamentals(self, symbol): return {}

def test_scan_isolates_failures_and_ranks(make_df, monkeypatch):
    import hub.run as run_mod
    monkeypatch.setattr(run_mod, "load_universe", lambda name: ["GOOD", "BAD"])
    frames = {"GOOD": make_df(list(range(100, 220)))}  # rising
    cfg = HubConfig.default()
    result = scan(cfg, StubProvider(frames))
    assert [c["symbol"] for c in result["candidates"]] == ["GOOD"]
    assert result["skipped"][0]["symbol"] == "BAD"

def test_write_reports_creates_files(tmp_path, make_df):
    cfg = HubConfig.default().__class__(out_dir=str(tmp_path))
    result = {"candidates": [{"symbol":"X","composite":0.5,"subscores":{},
                              "explanation":None}], "skipped": []}
    paths = write_reports(result, cfg, "20260628")
    assert all(__import__("os").path.exists(p) for p in paths.values())
