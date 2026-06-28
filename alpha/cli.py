import argparse, datetime as dt
from .config import AlphaConfig
from .panel import build_panel
from .backtest import run_backtest
from .report import write_report
from hub.data.provider import get_default_provider


def main(argv) -> int:
    p = argparse.ArgumentParser(prog="alpha")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--universe"); r.add_argument("--out")
    r.add_argument("--horizon", type=int); r.add_argument("--cost-bps", type=float)
    r.add_argument("--with-orthogonal", action="store_true")
    args = p.parse_args(argv)
    if args.cmd != "run":
        return 1
    cfg = AlphaConfig.default()
    over = {}
    if args.universe: over["universe"] = args.universe
    if args.out: over["out_dir"] = args.out
    if args.horizon: over["horizon"] = args.horizon
    if args.cost_bps is not None: over["cost_bps"] = args.cost_bps
    if over:
        cfg = AlphaConfig(**{**cfg.__dict__, **over})
    ext_provider = None
    if args.with_orthogonal:
        from .data import ExternalDataProvider
        cfg = AlphaConfig(**{**cfg.__dict__, "extra_features": ("rev_mom", "pead")})
        ext_provider = ExternalDataProvider(cfg.cache_dir + "_ext")
    provider = get_default_provider(cfg.cache_dir)
    panel = build_panel(provider, cfg, ext_provider=ext_provider)
    if len(panel) == 0:
        print("empty panel (no data fetched)"); return 1
    result = run_backtest(panel, cfg)
    date_str = dt.datetime.now().strftime("%Y%m%d")
    paths = write_report(result, panel, cfg, date_str)
    print(f"panel: {result['n_rows']} rows over {result['n_dates']} rebalances")
    print("Rank-IC (market+sector-neutral, OOS):")
    for name, s in result["ic"].items():
        verdict = "significant" if abs(s["t_stat"]) >= 2 else "noise"
        print(f"  {name:12s} IC={s['mean_ic']:+.4f}  t={s['t_stat']:+.2f}  ({verdict})")
    for name, v in result["portfolio"].items():
        print(f"  [{name}] Sharpe={v['sharpe']:.2f}  DSR={v['dsr']:.2f}  "
              f"({'PASS' if v['dsr']>=0.95 else 'not significant'})")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0
