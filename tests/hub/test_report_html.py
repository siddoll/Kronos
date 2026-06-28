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

def test_html_report_escapes_llm_content(tmp_path):
    # note/risk_flags derive from untrusted web news via the LLM — must be escaped.
    cfg = HubConfig(out_dir=str(tmp_path))
    result = {"candidates":[{"symbol":"X","composite":0.5,"subscores":{},
              "explanation":{"note":"<script>alert(1)</script>",
                             "risk_flags":["<img src=x onerror=alert(2)>"]}}],
              "skipped":[]}
    html = open(write_reports(result, cfg, "20260628")["html"]).read()
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(2)>" not in html
    assert "&lt;script&gt;" in html  # escaped form present
