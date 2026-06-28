import os
from hub.config import HubConfig
from hub.report import write_reports

def test_html_report_written_and_lists_symbol(tmp_path):
    cfg = HubConfig(out_dir=str(tmp_path))
    result = {"candidates":[{"symbol":"ZZZ","composite":0.77,
              "subscores":{"rvol":0.9},"explanation":{"note":"vol spike"}}],
              "skipped":[]}
    paths = write_reports(result, cfg, "20260628")
    assert "html" in paths and os.path.exists(paths["html"])
    html = open(paths["html"]).read()
    assert "ZZZ" in html and "vol spike" in html
