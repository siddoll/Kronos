import csv, os

_PATH = os.path.join(os.path.dirname(__file__), "data", "sectors.csv")

def load_sectors() -> dict:
    out = {}
    with open(_PATH) as f:
        for row in csv.DictReader(f):
            out[row["ticker"].strip()] = row["sector"].strip()
    return out
