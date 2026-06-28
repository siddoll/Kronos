import pandas as pd
from alpha.data import ExternalDataProvider as EP

def test_norm_ud_classifies_up_down():
    raw = pd.DataFrame(
        {"Action": ["up", "down", "main"], "priceTargetAction": ["", "", "Raises"]},
        index=pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]))
    raw.index.name = "GradeDate"
    out = EP._norm_ud(raw)
    assert list(out.columns) == ["date", "up", "down"]
    assert out["up"].sum() == 2 and out["down"].sum() == 1  # up, Raises -> up; down -> down

def test_norm_ud_handles_empty():
    out = EP._norm_ud(None)
    assert list(out.columns) == ["date", "up", "down"] and len(out) == 0

def test_norm_earn_extracts_surprise():
    raw = pd.DataFrame({"Surprise(%)": [3.5, -1.2]},
                       index=pd.to_datetime(["2025-01-30", "2024-10-30"]))
    raw.index.name = "Earnings Date"
    out = EP._norm_earn(raw)
    assert list(out.columns) == ["date", "surprise"]
    assert set(out["surprise"]) == {3.5, -1.2}
