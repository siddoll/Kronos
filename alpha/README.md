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

## Orthogonal features (Phase B)

    .venv/bin/python -m alpha run --with-orthogonal

Adds two point-in-time non-price features — analyst-revision momentum (`rev_mom`)
and earnings-drift/PEAD (`pead`, from yfinance; needs `lxml`) — and reports their
standalone Rank-IC plus a **price-only vs price+orthogonal** out-of-sample combiner
comparison (`lgbm_oos` vs `lgbm_oos_ext`).

> Short interest is snapshot-only on yfinance (no history) → not usable as a
> point-in-time feature; it needs a paid feed. The `ExternalDataProvider` interface
> is the seam to add paid revisions/short-interest/alt-data later.
