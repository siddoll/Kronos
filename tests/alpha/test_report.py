import os
from alpha.config import AlphaConfig
from alpha.report import write_report

def test_report_files_written(tmp_path, planted_panel):
    cfg = AlphaConfig(out_dir=str(tmp_path), n_folds=3, purge=1)
    result = {
        "ic": {"rvol": {"mean_ic": 0.08, "ic_ir": 0.5, "t_stat": 3.1, "n": 30},
               "composite": {"mean_ic": 0.06, "ic_ir": 0.4, "t_stat": 2.2, "n": 30}},
        "portfolio": {"composite": {"sharpe": 0.9, "sharpe_per_period": 0.26,
                                    "mean_net": 0.004, "dsr": 0.7}},
        "n_dates": 30, "n_rows": len(planted_panel)}
    paths = write_report(result, planted_panel, cfg, "20260628")
    for key in ("json", "csv", "html", "equity_png"):
        assert os.path.exists(paths[key])
    assert "rvol" in open(paths["html"]).read()
