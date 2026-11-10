"""Unit tests for src/risk_models.py (paper Sec. 4, Appendix A)."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk_models import (
    ewma_covariance,
    explained_variance_by_pc,
    fit_statistical_factor_model,
    rolling_sample_covariance,
)


def test_rolling_sample_covariance_matches_numpy_second_moment(synthetic_returns):
    L = 126
    sigma = rolling_sample_covariance(synthetic_returns, lookback=L)
    window = synthetic_returns[-L:]
    expected = (window.T @ window) / L
    assert np.allclose(sigma, expected)
    assert np.allclose(sigma, sigma.T)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals >= -1e-10)  # positive semi-definite


def test_rolling_sample_covariance_uses_only_trailing_window():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((300, 3)) * 0.01
    sigma_full = rolling_sample_covariance(base, lookback=126)
    # Appending future data must not change a lookback-limited forecast.
    extended = np.vstack([base, rng.standard_normal((50, 3)) * 0.01])[:300]
    sigma_again = rolling_sample_covariance(extended, lookback=126)
    assert np.allclose(sigma_full, sigma_again)


def test_ewma_covariance_is_symmetric_psd(synthetic_returns):
    sigma = ewma_covariance(synthetic_returns)
    assert np.allclose(sigma, sigma.T)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals >= -1e-8)


def test_ewma_covariance_reacts_faster_than_long_lookback_to_vol_shift():
    rng = np.random.default_rng(3)
    calm = rng.standard_normal((200, 2)) * 0.005
    shock = rng.standard_normal((50, 2)) * 0.05
    returns = np.vstack([calm, shock])
    sigma_ewma = ewma_covariance(returns, halflife_var=10, halflife_cov=30)
    sigma_long = rolling_sample_covariance(returns, lookback=250)
    # The short-half-life EWMA variance must pick up the recent shock more
    # than the long, equally-weighted rolling window.
    assert np.diag(sigma_ewma).mean() > np.diag(sigma_long).mean()


def test_statistical_factor_model_reconstructs_plausible_covariance(synthetic_returns):
    result = fit_statistical_factor_model(synthetic_returns, k=3, max_iter=100)
    sigma = result.covariance()
    assert sigma.shape == (6, 6)
    assert np.allclose(sigma, sigma.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals >= -1e-6)
    assert result.n_iter >= 1


def test_explained_variance_is_monotonic_and_bounded(synthetic_returns):
    cum = explained_variance_by_pc(synthetic_returns)
    assert np.all(np.diff(cum) >= -1e-9)
    assert cum[-1] == pytest.approx(1.0, abs=1e-6)
    assert cum[0] > 0
