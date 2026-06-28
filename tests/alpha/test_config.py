from alpha.config import AlphaConfig
from hub.config import SIGNAL_NAMES

def test_config_defaults():
    c = AlphaConfig.default()
    assert c.horizon == 21 and c.n_quantiles == 5
    assert set(c.weights) == set(SIGNAL_NAMES)

def test_fixtures_have_panel_schema(planted_panel):
    cols = set(planted_panel.columns)
    assert {"date","ticker","sector","fwd_ret"} <= cols
    assert set(SIGNAL_NAMES) <= cols
