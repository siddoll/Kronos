import argparse
import datetime as dt
from .config import HubConfig
from .data.provider import get_default_provider
from .run import scan
from .explain import explain_top
from .report import write_reports
from .universe import load_universe
from .screen.screener import run_screen
from .screen.library import get_preset

def main(argv) -> int:
    p = argparse.ArgumentParser(prog="hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--no-explain", action="store_true")
    s.add_argument("--top-k", type=int)
    s.add_argument("--universe")
    s.add_argument("--out")
    b = sub.add_parser("backtest")
    b.add_argument("--universe")
    b.add_argument("--horizon", type=int, default=10)
    sc = sub.add_parser("screen")
    sc.add_argument("--preset", required=True)
    sc.add_argument("--no-explain", action="store_true")
    sc.add_argument("--top-k", type=int)
    sc.add_argument("--universe")
    sc.add_argument("--out")
    args = p.parse_args(argv)

    if args.cmd == "scan":
        cfg = HubConfig.default()
        over = {}
        if args.top_k: over["top_k"] = args.top_k
        if args.universe: over["universe"] = args.universe
        if args.out: over["out_dir"] = args.out
        if over:
            cfg = HubConfig(**{**cfg.__dict__, **over})
        provider = get_default_provider(cfg.cache_dir)
        result = scan(cfg, provider)
        if not args.no_explain:
            import anthropic
            result = explain_top(result, provider, anthropic.Anthropic(), cfg)
        date_str = dt.datetime.now().strftime("%Y%m%d")
        paths = write_reports(result, cfg, date_str)
        print(f"{len(result['candidates'])} candidates, "
              f"{len(result['skipped'])} skipped")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
    if args.cmd == "screen":
        cfg = HubConfig.default()
        over = {}
        if args.top_k: over["top_k"] = args.top_k
        if args.universe: over["universe"] = args.universe
        if args.out: over["out_dir"] = args.out
        if over:
            cfg = HubConfig(**{**cfg.__dict__, **over})
        try:
            criteria = get_preset(args.preset)
        except ValueError as e:
            print(e); return 1
        provider = get_default_provider(cfg.cache_dir)
        universe = load_universe(cfg.universe)
        result = run_screen(universe, provider, criteria, top_k=cfg.top_k)
        if not args.no_explain and result["candidates"]:
            import anthropic
            result = explain_top(result, provider, anthropic.Anthropic(), cfg)
        date_str = dt.datetime.now().strftime("%Y%m%d")
        paths = write_reports(result, cfg, date_str)
        print(f"preset '{args.preset}': {len(result['candidates'])} passed, "
              f"{len(result['skipped'])} skipped")
        for c in result["candidates"][:10]:
            print(f"  {c['symbol']:6s} score={c['composite']:.2f}")
        for k, v in paths.items():
            print(f"  {k}: {v}")
        return 0
    if args.cmd == "backtest":
        from .validate import backtest_screen
        cfg = HubConfig.default()
        if args.universe:
            cfg = HubConfig(**{**cfg.__dict__, "universe": args.universe})
        # Separate cache dir: backtest needs more bars (lookback+horizon+) than a
        # scan caches, and the OHLCV cache is keyed by symbol only — sharing it
        # would shadow the longer fetch with a shorter scan frame.
        provider = get_default_provider(cfg.cache_dir + "_bt")
        need = cfg.lookback_days + args.horizon + 60
        frames = {}
        for sym in load_universe(cfg.universe):
            try:
                frames[sym] = provider.get_ohlcv(sym, need)
            except Exception:
                pass
        m = backtest_screen(frames, cfg, horizon=args.horizon)
        print("Walk-forward screen backtest (research funnel, not alpha):")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return 0
    return 1
