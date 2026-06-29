import pandas as pd
from hub.screen.library import PRESETS, get_preset, pe_below, growth_above
from hub.screen.criteria import near_52w_high

PRESET_NAMES = list(PRESETS)

def build_criteria(preset: str, overrides: dict) -> list:
    overrides = overrides or {}
    out = []
    for c in get_preset(preset):
        if c.name == "pe_below" and "pe_max" in overrides:
            out.append(pe_below(overrides["pe_max"]))
        elif c.name == "earnings_growth_above" and "eps_growth_min" in overrides:
            out.append(growth_above("earnings_growth", overrides["eps_growth_min"]))
        elif c.name == "near_52w_high" and "near_high_pct" in overrides:
            out.append(near_52w_high(overrides["near_high_pct"]))
        else:
            out.append(c)
    return out

def screen_to_table(result: dict) -> pd.DataFrame:
    rows = []
    for c in result.get("candidates", []):
        f = c.get("fundamentals") or {}
        crit = c.get("criteria") or {}
        passed = sum(1 for r in crit.values() if r.get("passed"))
        rows.append({
            "Symbol": c["symbol"],
            "Score": round(float(c["composite"]), 3),
            "Criteria": f"{passed}/{len(crit)}",
            "P/E": f.get("pe_ratio"),
            "EPS growth": f.get("earnings_growth"),
            "Net margin": f.get("net_margin"),
        })
    return pd.DataFrame(rows, columns=["Symbol", "Score", "Criteria", "P/E",
                                       "EPS growth", "Net margin"])


import json as _json


def load_screens(path) -> dict:
    try:
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return {}


def save_screen(name, settings, path) -> dict:
    screens = load_screens(path)
    screens[str(name)] = settings
    try:
        blob = _json.dumps(screens, indent=2)   # serialize before opening (no truncate-on-error)
        with open(path, "w") as f:
            f.write(blob)
    except Exception:
        pass
    return screens
