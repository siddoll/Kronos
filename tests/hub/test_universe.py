import pytest
from hub.universe import load_universe

def test_loads_sample_universe():
    syms = load_universe("sp500_sample")
    assert "AAPL" in syms and len(syms) >= 20
    assert all(s == s.strip() and not s.startswith("#") for s in syms)

def test_unknown_universe_raises():
    with pytest.raises(ValueError):
        load_universe("does_not_exist")
