import os
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest
from alpha.report import write_report

def test_end_to_end_offline(tmp_path, planted_panel):
    cfg = AlphaConfig(out_dir=str(tmp_path), n_folds=3, purge=1)
    result = run_backtest(planted_panel, cfg)
    paths = write_report(result, planted_panel, cfg, "20260628")
    assert os.path.exists(paths["html"]) and os.path.exists(paths["equity_png"])
    assert result["ic"]["rvol"]["mean_ic"] > 0.05

def test_cli_run_monkeypatched(tmp_path, planted_panel, monkeypatch):
    import alpha.cli as cli
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    monkeypatch.setattr(cli, "build_panel", lambda provider, cfg: planted_panel)
    rc = cli.main(["run", "--out", str(tmp_path)])
    assert rc == 0
    assert any(p.name.startswith("alpha_") for p in tmp_path.iterdir())
