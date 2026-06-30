from hub.ui.screen_runner import forward_test_table

def test_forward_table_labels_and_columns():
    ft = {"horizons": {5: {"hit_rate": 0.53, "pick_return": 0.012, "market_return": 0.010,
                           "edge": 0.002, "n": 120},
                       20: {"hit_rate": 0.55, "pick_return": 0.03, "market_return": 0.028,
                            "edge": 0.002, "n": 100}},
          "n_dates": 40, "n_names": 30}
    df = forward_test_table(ft)
    assert list(df.columns) == ["Horizon", "Picks up %", "Avg pick", "Avg market", "Edge", "n"]
    horizons = set(df["Horizon"])
    assert "1 week" in horizons and "4 weeks" in horizons
    assert df.iloc[0]["Picks up %"] == "53.0%"
