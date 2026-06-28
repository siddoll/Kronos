import pytest
from hub.screen.library import (pe_below, growth_above, margin_above, mktcap_above,
                                PRESETS, get_preset)

def test_fundamental_pass_fail_and_fail_closed():
    assert pe_below(40).evaluate(None, {"pe_ratio": 22.0}).passed
    assert not pe_below(40).evaluate(None, {"pe_ratio": 55.0}).passed
    assert not pe_below(40).evaluate(None, {"pe_ratio": None}).passed  # fail closed
    assert not pe_below(40).evaluate(None, {}).passed

def test_growth_and_margin():
    assert growth_above("earnings_growth", 0.1).evaluate(None, {"earnings_growth": 0.2}).passed
    assert not margin_above("net_margin", 0.1).evaluate(None, {"net_margin": 0.05}).passed

def test_presets_resolve():
    assert set(PRESETS) >= {"growth_momentum", "value", "quality_momentum"}
    crits = get_preset("growth_momentum")
    assert any(c.kind == "technical" for c in crits) and any(c.kind == "fundamental" for c in crits)
    with pytest.raises(ValueError):
        get_preset("nope")
