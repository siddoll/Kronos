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
    return {"json": json_path, "csv": csv_path}
