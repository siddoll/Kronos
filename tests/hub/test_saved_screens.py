from hub.ui.screen_runner import save_screen, load_screens

def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "screens.json")
    assert load_screens(p) == {}
    save_screen("my growth", {"preset": "growth_momentum", "pe_max": 30}, p)
    sc = load_screens(p)
    assert sc["my growth"]["pe_max"] == 30

def test_load_corrupt_is_empty(tmp_path):
    p = tmp_path / "bad.json"; p.write_text("{not json")
    assert load_screens(str(p)) == {}
