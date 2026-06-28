from hub.cli import main

def test_cli_scan_no_explain(tmp_path, monkeypatch):
    import hub.cli as cli
    # stub provider + universe so no network/LLM is used
    monkeypatch.setattr(cli, "get_default_provider", lambda d: None)
    def fake_scan(cfg, provider):
        return {"candidates":[{"symbol":"X","composite":0.5,"subscores":{},"explanation":None}],
                "skipped":[]}
    monkeypatch.setattr(cli, "scan", fake_scan)
    rc = main(["scan", "--no-explain", "--out", str(tmp_path)])
    assert rc == 0
    assert any(p.name.startswith("watchlist_") for p in tmp_path.iterdir())
