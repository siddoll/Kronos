from hub.cli import main

def test_screen_cli_runs(tmp_path, monkeypatch):
    import hub.cli as cli
    import numpy as np, pandas as pd
    def _price():
        n=300; c=np.linspace(80,160,n)
        return pd.DataFrame({"open":c,"high":c*1.01,"low":c*0.99,"close":c,"volume":1e6},
                            index=pd.date_range("2024-01-01", periods=n, freq="B"))
    class P:
        def get_ohlcv(self,s,l): return _price()
        def get_fundamentals(self,s): return {"pe_ratio":20.0,"earnings_growth":0.2,"net_margin":0.2}
        def get_news(self,s,limit=5): return []
    monkeypatch.setattr(cli, "get_default_provider", lambda d: P())
    monkeypatch.setattr(cli, "load_universe", lambda name: ["AAA","BBB"])
    rc = main(["screen", "--preset", "growth_momentum", "--no-explain", "--out", str(tmp_path)])
    assert rc == 0
    assert any(p.name.startswith("watchlist_") for p in tmp_path.iterdir())

def test_screen_cli_bad_preset(tmp_path, monkeypatch):
    import hub.cli as cli
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    monkeypatch.setattr(cli, "load_universe", lambda name: [])
    assert main(["screen", "--preset", "nope", "--out", str(tmp_path)]) == 1
