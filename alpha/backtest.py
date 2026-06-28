import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from hub.config import SIGNAL_NAMES
from .neutralize import neutralize
from .ic import rank_ic_series, ic_summary
from .splits import purged_walk_forward
from .combine import composite_score, linear_oos, lgbm_oos
from .portfolio import long_short
from .stats import deflated_sharpe_ratio


_EMPTY = {"ic": {}, "portfolio": {}, "n_dates": 0, "n_rows": 0}

def run_backtest(panel, cfg) -> dict:
    if panel is None or len(panel) == 0:
        return dict(_EMPTY)
    feats = list(SIGNAL_NAMES)                                  # 7 price signals (name kept for the score_map below)
    extras = [f for f in cfg.extra_features if f in panel.columns]
    resid = neutralize(panel).values
    dates = panel["date"].values
    folds = purged_walk_forward(dates, n_folds=cfg.n_folds, purge=cfg.purge)

    ic = {}
    for f in feats + extras:
        ic[f] = ic_summary(rank_ic_series(dates, panel[f].values, resid))
    comp = composite_score(panel, cfg.weights)
    ic["composite"] = ic_summary(rank_ic_series(dates, comp, resid))

    def _oos_ic(pred):
        m = ~np.isnan(pred)
        return ic_summary(rank_ic_series(dates[m], pred[m], resid[m]))

    lin = linear_oos(panel, resid, folds, feats)
    lgb = lgbm_oos(panel, resid, folds, feats)
    ic["linear_oos"] = _oos_ic(lin)
    ic["lgbm_oos"] = _oos_ic(lgb)
    if extras:
        feats_all = feats + extras
        ic["linear_oos_ext"] = _oos_ic(linear_oos(panel, resid, folds, feats_all))
        lgb = lgbm_oos(panel, resid, folds, feats_all)  # use the richer model for the portfolio
        ic["lgbm_oos_ext"] = _oos_ic(lgb)

    # One consistent trial set for the Deflated Sharpe: every config we evaluated
    # (7 signals + composite + linear_oos + lgbm_oos). sr_variance AND n_trials must
    # describe the SAME set, else DSR is biased (too few trials -> optimistic).
    ppy = max(1, round(252 / cfg.horizon))
    score_map = {f: panel[f].values for f in feats}
    score_map["composite"] = comp
    score_map["linear_oos"] = lin
    score_map["lgbm_oos"] = lgb
    runs = {name: long_short(panel, sc, cfg.n_quantiles, cfg.cost_bps,
                             periods_per_year=ppy)
            for name, sc in score_map.items()}
    spps = [r["sharpe_per_period"] for r in runs.values()]
    sr_var = float(np.var(spps)) or 1e-6
    n_trials = len(runs)  # same set the variance was measured over

    portfolio = {}
    for name in ("composite", "lgbm_oos"):
        r = runs[name]
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
