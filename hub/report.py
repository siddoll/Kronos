import os, json, csv

def write_reports(result: dict, cfg, date_str: str) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = os.path.join(cfg.out_dir, f"watchlist_{date_str}")
    json_path, csv_path = base + ".json", base + ".csv"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    cands = result["candidates"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "symbol", "composite", "explanation"])
        for i, c in enumerate(cands, 1):
            expl = (c.get("explanation") or {})
            note = expl.get("note", "") if isinstance(expl, dict) else ""
            w.writerow([i, c["symbol"], round(c["composite"], 4), note])
    paths = {"json": json_path, "csv": csv_path}
    paths["html"] = write_html(result, cfg, date_str)
    return paths


def write_html(result: dict, cfg, date_str: str) -> str:
    path = os.path.join(cfg.out_dir, f"report_{date_str}.html")
    rows = []
    for i, c in enumerate(result["candidates"], 1):
        expl = c.get("explanation") or {}
        note = expl.get("note", "") if isinstance(expl, dict) else ""
        flags = ", ".join((expl.get("risk_flags") or [])) if isinstance(expl, dict) else ""
        subs = " ".join(f"{k}:{v:.2f}" for k, v in c.get("subscores", {}).items())
        rows.append(f"<tr><td>{i}</td><td><b>{c['symbol']}</b></td>"
                    f"<td>{c['composite']:.3f}</td><td>{subs}</td>"
                    f"<td>{note}</td><td>{flags}</td></tr>")
    html = (f"<html><head><meta charset='utf-8'><title>Discovery Hub {date_str}</title>"
            "<style>body{font-family:sans-serif;margin:24px}"
            "table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #ddd;padding:6px;font-size:13px}"
            "th{background:#f4f4f4;text-align:left}</style></head><body>"
            f"<h2>Discovery Hub — {date_str}</h2>"
            f"<p>{len(result['candidates'])} candidates · {len(result['skipped'])} skipped. "
            "Research funnel, not buy signals.</p>"
            "<table><tr><th>#</th><th>Symbol</th><th>Score</th><th>Signals</th>"
            "<th>Why</th><th>Risk</th></tr>" + "".join(rows) + "</table></body></html>")
    with open(path, "w") as f:
        f.write(html)
    return path
