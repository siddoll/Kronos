import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from hub.config import SIGNAL_NAMES
from .neutralize import neutralize
from .ic import rank_ic_series, ic_summary
from .splits import purged_walk_forward
from .combine import composite_score, linear_oos, lgbm_oos
from .portfolio import long_short
from .stats import deflated_sharpe_ratio


def run_backtest(panel, cfg) -> dict:
    feats = list(SIGNAL_NAMES)
    resid = neutralize(panel).values
    dates = panel["date"].values
    folds = purged_walk_forward(dates, n_folds=cfg.n_folds, purge=cfg.purge)

    ic = {}
    for f in feats:
        ic[f] = ic_summary(rank_ic_series(dates, panel[f].values, resid))
    comp = composite_score(panel, cfg.weights)
    ic["composite"] = ic_summary(rank_ic_series(dates, comp, resid))
    lin = linear_oos(panel, resid, folds, feats)
    lgb = lgbm_oos(panel, resid, folds, feats)
    for name, pred in [("linear_oos", lin), ("lgbm_oos", lgb)]:
        m = ~np.isnan(pred)
        ic[name] = ic_summary(rank_ic_series(dates[m], pred[m], resid[m]))

    # portfolios for composite and the lgbm OOS predictions
    ppy = max(1, round(252 / cfg.horizon))
    scores = {"composite": comp, "lgbm_oos": lgb}
    spps = {}
    for name, sc in scores.items():
        spps[name] = long_short(panel, sc, cfg.n_quantiles, cfg.cost_bps,
                                periods_per_year=ppy)["sharpe_per_period"]
    sr_var = float(np.var(list(spps.values()))) or 1e-6
    n_trials = len(ic)  # configs evaluated
    portfolio = {}
    for name, sc in scores.items():
        r = long_short(panel, sc, cfg.n_quantiles, cfg.cost_bps, periods_per_year=ppy)
        net = r["net"].dropna()
        dsr = deflated_sharpe_ratio(
            r["sharpe_per_period"], len(net),
            float(_skew(net)) if len(net) > 2 else 0.0,
            float(_kurtosis(net, fisher=False)) if len(net) > 3 else 3.0,
            sr_var, n_trials)
        portfolio[name] = {"sharpe": r["sharpe"],
                           "sharpe_per_period": r["sharpe_per_period"],
                           "mean_net": float(net.mean()), "dsr": dsr}
    return {"ic": ic, "portfolio": portfolio,
            "n_dates": len(set(panel["date"].astype(str))), "n_rows": len(panel)}
