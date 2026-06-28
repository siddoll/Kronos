import pandas as pd
from alpha.splits import purged_walk_forward

def test_folds_are_oos_and_purged():
    dates = pd.date_range("2022-01-01", periods=50, freq="21D")
    folds = purged_walk_forward(dates, n_folds=4, purge=2)
    assert len(folds) == 4
    for train, test in folds:
        assert max(train) < min(test)                       # strictly out-of-sample
        # purge gap: at least `purge` rebalance dates between last train and first test
        all_d = sorted(pd.to_datetime(dates).unique())
        gap = all_d.index(min(test)) - all_d.index(max(train))
        assert gap >= 2

def test_expanding_train_grows():
    dates = pd.date_range("2022-01-01", periods=50, freq="21D")
    folds = purged_walk_forward(dates, n_folds=4, purge=1)
    assert len(folds[1][0]) > len(folds[0][0])  # train expands
