import pytest
from alpha.universe import load_universe
from alpha.config import AlphaConfig

def test_default_universe_loads_from_alpha_data():
    # regression for C1: the default `alpha run` universe must resolve (it lives
    # in alpha/data/, not hub/data/) — otherwise the live CLI crashes.
    syms = load_universe(AlphaConfig.default().universe)
    assert "AAPL" in syms and len(syms) >= 50

def test_unknown_universe_raises():
    with pytest.raises(ValueError):
        load_universe("nope")
