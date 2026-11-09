# Modified for METAN portable release 2026-11-09.
"""Original-unit regression metrics; undefined correlations/R2 are NaN, never zero."""
import math
import statistics


def regression_metrics(y_true, y_pred):
    y = [float(x) for x in y_true]
    p = [float(x) for x in y_pred]
    if len(y) != len(p) or not y:
        raise ValueError('Aligned nonempty vectors required')
    if not all(math.isfinite(x) for x in y+p):
        raise ValueError('Nonfinite prediction/label; do not silently drop records')
    n = len(y)
    ym, pm = statistics.mean(y), statistics.mean(p)
    ss = sum((v-ym)**2 for v in y)
    ps = sum((v-pm)**2 for v in p)
    squared = sum((u-v)**2 for u,v in zip(y,p))
    mse = squared/n
    pearson = sum((u-ym)*(v-pm) for u,v in zip(y,p))/math.sqrt(ss*ps) if ss > 0 and ps > 0 else math.nan
    return {'n': n, 'Pearson': pearson, 'MSE': mse, 'RMSE': math.sqrt(mse),
            'MAE': sum(abs(u-v) for u,v in zip(y,p))/n,
            'R2': 1-squared/ss if ss > 0 else math.nan,
            'prediction_SD_ratio': math.sqrt(ps/ss) if ss > 0 and n > 1 else math.nan}


def mean_sd(values):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('All requested folds must contain finite valid metrics')
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else math.nan
