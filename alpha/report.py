import os, json, csv
import html as _html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .combine import composite_score
from .portfolio import long_short


def write_report(result, panel, cfg, date_str) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = os.path.join(cfg.out_dir, f"alpha_{date_str}")
    paths = {"json": base + ".json", "csv": base + ".csv", "html": base + ".html",
             "equity_png": os.path.join(cfg.out_dir, f"equity_{date_str}.png")}

    with open(paths["json"], "w") as f:
        json.dump(result, f, indent=2, default=str)

    ic = result["ic"]
    with open(paths["csv"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_ic", "ic_ir", "t_stat", "n"])
        for name, s in ic.items():
            w.writerow([name, round(s["mean_ic"], 4), round(s["ic_ir"], 3),
                        round(s["t_stat"], 2), s["n"]])

    # equity curve for composite
    try:
        eq = long_short(panel, composite_score(panel, cfg.weights),
                        cfg.n_quantiles, cfg.cost_bps)["equity"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(eq.index, eq.values, color="#1f4eb0")
        ax.set_title("Composite long-short equity (after costs)")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(paths["equity_png"], dpi=110); plt.close(fig)
    except Exception:
        # still emit an (empty) file so the path exists
        plt.figure(); plt.savefig(paths["equity_png"]); plt.close()

    rows = "".join(
        f"<tr><td>{_html.escape(n)}</td><td>{s['mean_ic']:.4f}</td>"
        f"<td>{s['ic_ir']:.3f}</td><td>{s['t_stat']:.2f}</td><td>{s['n']}</td>"
        f"<td>{'significant' if abs(s['t_stat'])>=2 else '— noise'}</td></tr>"
        for n, s in ic.items())
    pf = result.get("portfolio", {})
    pf_rows = "".join(
        f"<tr><td>{_html.escape(k)}</td><td>{v['sharpe']:.2f}</td>"
        f"<td>{v['mean_net']:.4f}</td><td>{v['dsr']:.2f}</td>"
        f"<td>{'PASS' if v['dsr']>=0.95 else 'fail'}</td></tr>"
        for k, v in pf.items())
    html = (f"<html><head><meta charset='utf-8'><title>Alpha Engine {date_str}</title>"
            "<style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;margin-bottom:20px}"
            "td,th{border:1px solid #ddd;padding:6px;font-size:13px}"
            "th{background:#f4f4f4}</style></head><body>"
            f"<h2>Alpha Engine — {date_str}</h2>"
            f"<p>{result['n_rows']} name-dates over {result['n_dates']} rebalances. "
            "Market+sector-neutral target, out-of-sample, after costs. "
            "Significance: |t|&#x2265;2 for IC; Deflated Sharpe &#x2265;0.95 for the portfolio.</p>"
            "<h3>Rank-IC</h3><table><tr><th>Model</th><th>Mean IC</th><th>IC-IR</th>"
            f"<th>t-stat</th><th>n</th><th>verdict</th></tr>{rows}</table>"
            "<h3>Long-short portfolio</h3><table><tr><th>Model</th><th>Sharpe</th>"
            f"<th>Mean net</th><th>DSR</th><th>verdict</th></tr>{pf_rows}</table>"
            f"<p><img src='{os.path.basename(paths['equity_png'])}' width='720'></p>"
            "</body></html>")
    with open(paths["html"], "w") as f:
        f.write(html)
    return paths
