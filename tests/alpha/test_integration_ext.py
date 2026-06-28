import numpy as np
import pandas as pd
from alpha.config import AlphaConfig
from alpha.backtest import run_backtest
from alpha.report import write_report
from hub.config import SIGNAL_NAMES

def _panel(seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=40, freq="21D")
    tickers = [f"T{i:02d}" for i in range(30)]
    sectors = {t: ["A", "B", "C"][i % 3] for i, t in enumerate(tickers)}
    rows = []
    for d in dates:
        sec = {s: rng.normal(0, 0.05) for s in set(sectors.values())}
        mkt = rng.normal(0, 0.03)
        for t in tickers:
            feats = {n: rng.normal() for n in SIGNAL_NAMES}
            rev = rng.normal()
            rows.append({"date": d, "ticker": t, "sector": sectors[t], **feats,
                         "rev_mom": rev, "pead": rng.normal(),
                         "fwd_ret": mkt + sec[sectors[t]] + 0.05*rev + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)

def test_ext_end_to_end_report(tmp_path):
    cfg = AlphaConfig(out_dir=str(tmp_path), n_folds=3, purge=1,
                      extra_features=("rev_mom", "pead"))
    result = run_backtest(_panel(), cfg)
    paths = write_report(result, _panel(), cfg, "20260628")
    html = open(paths["html"]).read()
    assert "rev_mom" in html and "lgbm_oos_ext" in html

def test_cli_with_orthogonal_monkeypatched(tmp_path, monkeypatch):
    import alpha.cli as cli
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    monkeypatch.setattr(cli, "build_panel",
                        lambda provider, cfg, ext_provider=None: _panel())
    rc = cli.main(["run", "--with-orthogonal", "--out", str(tmp_path)])
    assert rc == 0
