import numpy as np
import pandas as pd

def purged_walk_forward(dates, n_folds: int = 4, purge: int = 2):
    udates = sorted(pd.to_datetime(pd.Series(dates)).unique())
    n = len(udates)
    fold = n // (n_folds + 1)
    if fold == 0:
        return []
    out = []
    for i in range(1, n_folds + 1):
        test_start = i * fold
        test_end = (i + 1) * fold if i < n_folds else n
        test = udates[test_start:test_end]
        train_end = max(0, test_start - purge)
        train = udates[:train_end]
        if train and test:
            out.append(([pd.Timestamp(d) for d in train],
                        [pd.Timestamp(d) for d in test]))
    return out
