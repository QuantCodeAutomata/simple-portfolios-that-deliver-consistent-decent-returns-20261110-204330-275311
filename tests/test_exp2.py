"""Unit tests for exp/exp2.py (paper Sec. 4, Sec. 6.3, Table 17/18).

All tests use synthetic numpy data; no network calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from exp.exp2 import (
    eligible_tickers_for_period,
    frobenius_distance,
    tickers_of_asset_class,
    valid_forecast_positions,
)
from src.risk_models import ewma_covariance, fit_statistical_factor_model, rolling_sample_covariance
from src.universe import ASSET_CLASS_OF_CATEGORY, CATEGORIES, FULL_UNIVERSE


def test_frobenius_distance_is_zero_for_a_perfect_forecast():
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((21, 5)) * 0.01
    sigma_real = (returns.T @ returns) / 21
    assert frobenius_distance(sigma_real, sigma_real) == pytest.approx(0.0, abs=1e-12)


def test_frobenius_distance_is_positive_for_an_imperfect_forecast():
    rng = np.random.default_rng(2)
    sigma_real = np.eye(4) * 0.0004
    sigma_hat = sigma_real + np.eye(4) * 0.0001
    d = frobenius_distance(sigma_hat, sigma_real)
    assert d > 0.0


@pytest.fixture
def synthetic_returns_300x5() -> np.ndarray:
    rng = np.random.default_rng(7)
    n_days, n_assets = 300, 5
    vols = np.array([0.01, 0.015, 0.008, 0.02, 0.012])
    corr = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, n_assets))
    return (z @ chol.T) * vols


@pytest.mark.parametrize(
    "estimator",
    [
        lambda r: rolling_sample_covariance(r, lookback=126),
        lambda r: ewma_covariance(r),
        lambda r: fit_statistical_factor_model(r, k=3, max_iter=50).covariance(),
    ],
    ids=["rolling_sample", "ewma", "factor_model_k3"],
)
def test_covariance_estimators_return_symmetric_psd_matrices(synthetic_returns_300x5, estimator):
    sigma = estimator(synthetic_returns_300x5)
    n_assets = synthetic_returns_300x5.shape[1]
    assert sigma.shape == (n_assets, n_assets)
    assert np.allclose(sigma, sigma.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals >= -1e-8)


def test_tickers_of_asset_class_partitions_categories_without_overlap():
    equities = tickers_of_asset_class("Equities")
    fixed_income = tickers_of_asset_class("Fixed income")
    alternatives = tickers_of_asset_class("Alternatives")
    assert set(equities) & set(fixed_income) == set()
    assert set(equities) & set(alternatives) == set()
    assert set(fixed_income) & set(alternatives) == set()
    assert sorted(equities + fixed_income + alternatives) == sorted(FULL_UNIVERSE)
    for cat, tickers in CATEGORIES.items():
        expected_class = ASSET_CLASS_OF_CATEGORY[cat]
        target = {"Equities": equities, "Fixed income": fixed_income, "Alternatives": alternatives}[expected_class]
        assert set(tickers) <= set(target)


def test_valid_forecast_positions_respects_history_and_horizon_bounds():
    # n_dates small enough that only a narrow band of rebalance positions qualifies
    rebalance_positions = [100, 200, 251, 252, 300, 400, 479, 480]
    n_dates = 500
    valid = valid_forecast_positions(n_dates, rebalance_positions)
    # lo = 126 + 126 = 252; hi = 500 - 21 = 479
    assert valid == [252, 300, 400, 479]


def test_eligible_tickers_for_period_drops_late_listed_tickers():
    listing_index = {"OLD1": 0, "OLD2": 10, "NEW": 400}
    first_eval_t = 252
    eligible = eligible_tickers_for_period(["OLD1", "OLD2", "NEW", "MISSING"], listing_index, first_eval_t)
    # NEW: 400 + 126 > 252 -> excluded; MISSING has no listing_index -> excluded
    assert eligible == ["OLD1", "OLD2"]
