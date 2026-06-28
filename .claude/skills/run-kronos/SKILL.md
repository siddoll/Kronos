---
name: run-kronos
description: Use when asked to run, launch, demo, or smoke-test Kronos in this repo — either a forecast from the example/test data or the Flask Web UI. Captures the verified Python 3.13 setup (incl. the pandas pin that segfaults) so it works first try.
---

# Running Kronos locally

Kronos is a decoder-only foundation model for financial K-lines. Two ways to "run" it:
a **headless forecast script**, or the **Flask Web UI** on port 7070. Both download
model weights from Hugging Face on first use (`NeoQuasar/Kronos-*`).

## 0. The one gotcha (read this first)

`requirements.txt` historically pinned `pandas==2.2.2`, which has **no cp313 wheel**.
On a Python 3.13 venv, pip builds it from source against numpy 2.x and the result
**segfaults (exit 139) inside `pd.to_datetime`** — looks like a model bug but is a
binary/ABI mismatch, and it crashes before any model code runs. The fix is already in
the repo (pinned to `pandas>=2.2.3`). If you ever see that segfault, the pandas pin is
the cause: `pip install --only-binary=:all: "pandas>=2.2.3"`.

## 1. Build the environment (once)

```bash
cd <repo-root>
python3.13 -m venv .venv          # 3.14 is too new for stable torch wheels; 3.13 works
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
# verify
.venv/bin/python -c "import torch, pandas; from model import Kronos, KronosTokenizer, KronosPredictor; print('ok', torch.__version__, pandas.__version__)"
```

## 2. Run a headless forecast

The official `examples/prediction_example.py` references a **missing** data file
(`examples/data/XSHG_5min_600977.csv`) and calls `plt.show()` (no display in a
sandbox). Use `tests/data/regression_input.csv` instead — it has the exact expected
columns (`timestamps,open,high,low,close,volume,amount`, 2500 rows). A ready headless
runner that loads it, runs `KronosPredictor`, and saves a PNG lives at
`.context/run_forecast.py`:

```bash
.venv/bin/python .context/run_forecast.py   # -> .context/forecast_result.png
```

Drive-it check: read the PNG. Real success = blue ground-truth close/volume with a red
prediction overlaid over the last 120 of 520 points. A blank frame is a failure.

## 3. Run the Web UI (Flask, port 7070)

The Web UI needs `flask flask-cors plotly` (see `webui/requirements.txt`; its pandas
pin is fixed too). The UI scans `<repo-root>/data/` for `.csv`/`.feather`, so seed it.
`webui/app.py` hardcodes `debug=True` (forking reloader — bad for background launch),
so launch via the wrapper at `.context/run_webui.py` which runs `use_reloader=False`.

```bash
.venv/bin/pip install flask flask-cors plotly          # do NOT reinstall webui/requirements.txt verbatim — it re-pins broken pandas
mkdir -p data && cp tests/data/regression_input.csv data/sample.csv
nohup .venv/bin/python -u .context/run_webui.py > .context/webui.log 2>&1 &
```

Smoke-test the API (this is the exact path the frontend calls):

```bash
curl -s http://127.0.0.1:7070/api/model-status
curl -s http://127.0.0.1:7070/api/data-files
# full money path: load a model, then predict
curl -s -X POST http://127.0.0.1:7070/api/load-model -H 'Content-Type: application/json' \
  -d '{"model_key":"kronos-small","device":"cpu"}'
curl -s -X POST http://127.0.0.1:7070/api/predict -H 'Content-Type: application/json' \
  -d '{"file_path":"<repo-root>/data/sample.csv","lookback":400,"pred_len":120,"temperature":1.0,"top_p":0.9,"sample_count":1}'
```

A working predict returns `{"success": true, ...}` with 120 `prediction_results` and a
non-empty `chart`. In the browser at http://127.0.0.1:7070 the user drives it: pick
model + device → Load Model → pick data file → Load Data → Start Prediction.

## Notes

- Devices: `cpu` (safe), `mps` (Apple Silicon GPU), `cuda` (NVIDIA).
- Window is fixed at 400 lookback + 120 prediction = 520 points; the data file needs ≥520 rows.
- `.context/` and `data/` are local scratch — keep them out of commits.
