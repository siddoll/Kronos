import argparse
import datetime as dt
from .config import HubConfig
from .data.provider import get_default_provider
from .run import scan
from .explain import explain_top
from .report import write_reports

def main(argv) -> int:
    p = argparse.ArgumentParser(prog="hub")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--no-explain", action="store_true")
    s.add_argument("--top-k", type=int)
    s.add_argument("--universe")
    s.add_argument("--out")
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
    return 1
