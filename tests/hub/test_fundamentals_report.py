import os
from hub.config import HubConfig
from hub.report import write_reports

def test_html_shows_fundamentals(tmp_path):
    cfg = HubConfig(out_dir=str(tmp_path))
    result = {"candidates": [{"symbol": "AAPL", "composite": 0.7, "subscores": {},
              "explanation": {"note": "n"},
              "fundamentals": {"pe_ratio": 30.5, "earnings_growth": 0.12,
                               "net_margin": 0.25}}], "skipped": []}
    html = open(write_reports(result, cfg, "20260628")["html"]).read()
    assert "30.5" in html and "P/E" in html
