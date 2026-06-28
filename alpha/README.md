# Alpha Engine (Phase A)

Cross-sectional, factor-neutral backtest of the Discovery Hub's signals — the honest
test of whether they hold tradable alpha after neutralization, costs, and
multiple-testing correction.

> Research/measurement harness. Not trading, not advice. Expect IC ~0.02-0.05
> at best on this liquid universe; a bigger number usually means overfitting.

## Run

    .venv/bin/pip install -r alpha/requirements.txt
    .venv/bin/python -m alpha run                 # full live backtest (yfinance)
    .venv/bin/python -m alpha run --cost-bps 5    # override costs

Output in `out_alpha/`: IC table (CSV/JSON), HTML report, equity-curve PNG.
Reports Rank-IC (mean, IC-IR, t-stat) per signal + composite + linear/LightGBM
out-of-sample, and the long-short Sharpe with **Deflated Sharpe Ratio**.
