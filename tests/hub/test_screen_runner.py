from hub.ui.screen_runner import build_criteria, screen_to_table, PRESET_NAMES

def test_preset_names():
    assert "growth_momentum" in PRESET_NAMES and "value" in PRESET_NAMES

def test_build_criteria_overrides_pe():
    crits = build_criteria("growth_momentum", {"pe_max": 18})
    pe = next(c for c in crits if c.name == "pe_below")
    assert pe.evaluate(None, {"pe_ratio": 15.0}).passed       # 15 < 18 -> pass
    assert not pe.evaluate(None, {"pe_ratio": 25.0}).passed    # 25 < 18 -> fail

def test_build_criteria_ignores_irrelevant_override():
    # 'value' has no near_52w_high; overriding it must not error or add it
    crits = build_criteria("value", {"near_high_pct": 0.03})
    assert all(c.name != "near_52w_high" for c in crits)

def test_screen_to_table_shape_and_none_safe():
    result = {"candidates": [
        {"symbol": "AAA", "composite": 0.71,
         "criteria": {"pe_below": {"passed": True, "value": 20, "score": 1.0},
                      "near_52w_high": {"passed": False, "value": 100, "score": 0.0}},
         "fundamentals": {"pe_ratio": 20.0, "earnings_growth": 0.1, "net_margin": None}}],
        "skipped": []}
    df = screen_to_table(result)
    assert list(df.columns) == ["Symbol", "Score", "Criteria", "P/E", "EPS growth", "Net margin"]
    assert df.iloc[0]["Symbol"] == "AAA" and df.iloc[0]["Criteria"] == "1/2"
