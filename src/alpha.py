"""Alpha (return-forecast) models (paper Sec. 5.2, 5.3).

Simple signals (one-month momentum, 50-day moving average) and a per-asset
rolling ridge regression on non-overlapping price-based features (STR, MTM,
LTR). Context7 (scikit-learn Ridge) confirmed that ``sklearn.linear_model
.Ridge.fit`` accepts a ``sample_weight`` array, so the half-life-weighted
training scheme of Table 19 is implemented with scikit-learn's Ridge rather
than a hand-rolled normal-equations solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import (
    LTR_WINDOW,
    MOMENTUM_LOOKBACK,
    MOVAVG_LOOKBACK,
    MOVAVG_RHO,
    MTM_WINDOW,
    RIDGE_FORECAST_HORIZON,
    RIDGE_HALFLIFE,
    RIDGE_LOOKBACK_H,
    STR_WINDOW,
)


def momentum_signal(prices: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK) -> pd.DataFrame:
    """One-month momentum alpha: log(p_t / p_{t-lookback}) (Sec. 5.2)."""
    return np.log(prices / prices.shift(lookback))


def moving_average_signal(prices: pd.DataFrame, lookback: int = MOVAVG_LOOKBACK, rho: float = MOVAVG_RHO) -> pd.DataFrame:
    """+1 when p_t > rho * mean(p_{t-lookback..t-1}), else -1 (Sec. 5.2)."""
    trailing_avg = prices.shift(1).rolling(lookback).mean()
    return np.where(prices > rho * trailing_avg, 1.0, -1.0)


def _non_overlapping_features(prices: pd.Series) -> pd.DataFrame:
    """STR, MTM, LTR features on non-overlapping historical windows (Sec. 5.3).

    STR_t = -log(p_{t-1}/p_{t-21})
    MTM_t = log(p_{t-21}/p_{t-252})
    LTR_t = -log(p_{t-252}/p_{t-756})
    """
    str_ = -np.log(prices.shift(STR_WINDOW[0]) / prices.shift(STR_WINDOW[1]))
    mtm = np.log(prices.shift(MTM_WINDOW[0]) / prices.shift(MTM_WINDOW[1]))
    ltr = -np.log(prices.shift(LTR_WINDOW[0]) / prices.shift(LTR_WINDOW[1]))
    return pd.DataFrame({"STR": str_, "MTM": mtm, "LTR": ltr})


def _sample_weights(n: int, halflife: float) -> np.ndarray:
    lam = np.exp(np.log(0.5) / halflife)
    age = np.arange(n - 1, -1, -1)
    return lam ** age


def rolling_ridge_oos_r2(
    prices: pd.Series,
    lam: float,
    lookback_h: int = RIDGE_LOOKBACK_H,
    horizon: int = RIDGE_FORECAST_HORIZON,
    halflife: float = RIDGE_HALFLIFE,
) -> float:
    """Out-of-sample R^2 of a rolling ridge regression for one asset (Sec. 5.3).

    At each forecast date t (row index i), the target y_{t'} = r_{t'+horizon}
    of a training pair is only known once t'+horizon has passed, so a fold
    may only use pairs "ending at t - h - 1" (Sec. 5.3 / Table 19's training
    scheme): the most recent complete pair has training index t - horizon,
    i.e. the training slice is ``x[i-horizon-lookback_h : i-horizon]``. An
    earlier version of this function trained on ``x[i-lookback_h:i]``
    instead, which includes targets y_{t'} for t' up to i-1 that embed prices
    up to idx[i-1]+horizon — horizon-1 trading days *after* the forecast date
    idx[i] — a look-ahead leak that inflated the out-of-sample R^2 to
    positive values, contradicting the paper's reported negative R^2 at every
    regularization level.
    """
    features = _non_overlapping_features(prices)
    fwd_return = np.log(prices.shift(-horizon) / prices)
    data = pd.concat([features, fwd_return.rename("target")], axis=1).dropna()
    assert len(data) > lookback_h + 2 * horizon, "not enough history for a single ridge fold"

    x = data[["STR", "MTM", "LTR"]].to_numpy()
    y = data["target"].to_numpy()

    num, den = 0.0, 0.0
    n_folds = 0
    sw = _sample_weights(lookback_h, halflife)
    for i in range(lookback_h + horizon, len(data) - horizon):
        train_end = i - horizon  # exclusive: last training pair ends at t - h - 1
        train_x = x[train_end - lookback_h:train_end]
        train_y = y[train_end - lookback_h:train_end]
        model = Ridge(alpha=lam, fit_intercept=True)
        model.fit(train_x, train_y, sample_weight=sw)
        y_hat = model.predict(x[i:i + 1])[0]
        y_true = y[i]
        num += (y_true - y_hat) ** 2
        den += y_true ** 2
        n_folds += 1
    assert n_folds > 0, "no out-of-sample folds produced"
    return float(1.0 - num / den)
