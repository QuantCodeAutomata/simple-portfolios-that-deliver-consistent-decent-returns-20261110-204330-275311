"""Unit tests for src/alpha.py (paper Sec. 5.2, 5.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.alpha import momentum_signal, moving_average_signal, rolling_ridge_oos_r2


def _geometric_price_path(n: int, drift: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)


def test_momentum_signal_positive_for_uptrend():
    prices = _geometric_price_path(300, drift=0.001, seed=0).to_frame("A")
    mom = momentum_signal(prices, lookback=21)
    assert mom["A"].dropna().mean() > 0


def test_momentum_signal_matches_log_return_definition():
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105]})
    mom = momentum_signal(prices, lookback=2)
    expected = np.log(104 / 102)
    assert mom["A"].iloc[4] == pytest.approx(expected)


def test_moving_average_signal_is_plus_minus_one():
    prices = _geometric_price_path(200, drift=0.0005, seed=1).to_frame("A")
    signal = moving_average_signal(prices, lookback=50, rho=1.0)
    values = np.unique(signal[~np.isnan(signal)])
    assert set(values.tolist()) <= {-1.0, 1.0}


def test_moving_average_signal_flags_uptrend_as_long():
    # A strictly increasing price path should end up above its trailing mean.
    idx = pd.bdate_range("2010-01-01", periods=120)
    prices = pd.DataFrame({"A": np.linspace(100, 200, 120)}, index=idx)
    signal = moving_average_signal(prices, lookback=50, rho=1.0)
    assert signal[-1, 0] == 1.0


def test_rolling_ridge_oos_r2_is_finite_and_bounded_above_by_one():
    prices = _geometric_price_path(1200, drift=0.0002, seed=2)
    r2 = rolling_ridge_oos_r2(prices, lam=1.0, lookback_h=100, horizon=21, halflife=100)
    assert np.isfinite(r2)
    assert r2 <= 1.0


def test_ridge_coefficients_shrink_towards_zero_as_lambda_grows():
    # A basic mathematical property of ridge regression (Sec. 5.3): the L2 norm
    # of the fitted coefficients is non-increasing in the penalty lambda.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 3))
    y = x @ np.array([0.3, -0.2, 0.1]) + rng.normal(scale=0.5, size=200)
    norms = []
    for lam in (1e-3, 1.0, 1e2, 1e4, 1e6):
        model = Ridge(alpha=lam, fit_intercept=True).fit(x, y)
        norms.append(np.linalg.norm(model.coef_))
    assert all(a >= b - 1e-9 for a, b in zip(norms, norms[1:]))
    assert norms[-1] < norms[0]
