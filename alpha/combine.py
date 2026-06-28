import numpy as np
import pandas as pd


def composite_score(panel: pd.DataFrame, weights: dict) -> np.ndarray:
    feats = list(weights)

    def _z(g):
        return (g - g.mean()) / (g.std(ddof=0) + 1e-9)

    z = panel.groupby("date")[feats].transform(_z)
    score = sum(z[f].values * weights[f] for f in feats)
    return np.asarray(score, dtype=float)


def _oos(panel, target, folds, feats, fit_predict):
    pred = np.full(len(panel), np.nan)
    date = panel["date"].values.astype("datetime64[ns]")
    X = panel[feats].values.astype(float)
    y = np.asarray(target, float)
    for train_dates, test_dates in folds:
        tr = np.isin(date, np.array(train_dates, dtype="datetime64[ns]"))
        te = np.isin(date, np.array(test_dates, dtype="datetime64[ns]"))
        if tr.sum() < 10 or te.sum() == 0:
            continue
        med = np.nanmedian(X[tr], axis=0)          # train-fold medians (no leakage)
        med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        Xte = np.where(np.isnan(X[te]), med, X[te])
        pred[te] = fit_predict(Xtr, y[tr], Xte)
    return pred


def linear_oos(panel, target, folds, feats) -> np.ndarray:
    def fp(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
        return B @ coef

    return _oos(panel, target, folds, feats, fp)


def lgbm_oos(panel, target, folds, feats) -> np.ndarray:
    import lightgbm as lgb

    def fp(Xtr, ytr, Xte):
        m = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
        m.fit(Xtr, ytr)
        return m.predict(Xte)

    return _oos(panel, target, folds, feats, fp)
