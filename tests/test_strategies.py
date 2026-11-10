"""Unit tests for src/strategies.py (paper Sec. 3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategies import (
    fixed_weights,
    inverse_volatility,
    maximum_diversification,
    minimum_volatility,
    risk_parity,
    volatility_control,
)


def _cov_from_vols_corr(vols: np.ndarray, rho: float) -> np.ndarray:
    n = len(vols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vols, vols) * corr


def test_fixed_weights_rejects_shorting_and_leverage():
    assert np.allclose(fixed_weights(np.array([0.5, 0.5])), [0.5, 0.5])
    with pytest.raises(AssertionError):
        fixed_weights(np.array([-0.1, 1.1]))
    with pytest.raises(AssertionError):
        fixed_weights(np.array([0.7, 0.7]))


def test_volatility_control_scales_down_high_vol_portfolio():
    w = np.array([0.5, 0.5])
    cov_high = _cov_from_vols_corr(np.array([0.30, 0.30]), 0.3)  # 30% annual vol-ish daily proxy
    scaled = volatility_control(w, cov_high, sigma_tgt=0.08)
    assert scaled.sum() < w.sum()
    assert np.all(scaled >= 0)


def test_volatility_control_capped_at_one_for_low_vol():
    w = np.array([0.3, 0.3])
    cov_tiny = _cov_from_vols_corr(np.array([1e-5, 1e-5]), 0.0)
    scaled = volatility_control(w, cov_tiny, sigma_tgt=0.08)
    assert np.allclose(scaled, w)  # theta capped at 1, no scale-up


def test_inverse_volatility_assigns_more_to_lower_vol_asset():
    cov = _cov_from_vols_corr(np.array([0.1, 0.4]), 0.0)
    w = inverse_volatility(cov)
    assert w[0] > w[1]
    assert w.sum() == pytest.approx(1.0)


def test_minimum_volatility_sums_to_one_and_nonnegative():
    cov = _cov_from_vols_corr(np.array([0.1, 0.2, 0.15]), 0.2)
    w = minimum_volatility(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert np.all(w >= -1e-6)


def test_minimum_volatility_concentrates_in_lowest_vol_asset():
    cov = _cov_from_vols_corr(np.array([0.05, 0.30, 0.30]), 0.3)
    w = minimum_volatility(cov)
    assert w[0] > w[1] and w[0] > w[2]


def test_risk_parity_equalizes_marginal_risk_contributions():
    cov = _cov_from_vols_corr(np.array([0.1, 0.2, 0.3]), 0.2)
    w = risk_parity(cov)
    mrc = cov @ w * w  # risk contribution_i = w_i * (Sigma w)_i
    mrc = mrc[w > 1e-6]
    assert np.allclose(mrc, mrc.mean(), rtol=0.05)


def test_risk_parity_admits_small_allocation_to_high_vol_asset():
    # Risk parity should never fully exclude an asset, unlike min-vol.
    cov = _cov_from_vols_corr(np.array([0.05, 0.05, 0.60]), 0.1)
    w = risk_parity(cov)
    assert w[2] > 1e-4


def test_maximum_diversification_sums_to_one_and_nonnegative():
    cov = _cov_from_vols_corr(np.array([0.1, 0.2, 0.15, 0.25]), 0.1)
    w = maximum_diversification(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-4)
    assert np.all(w >= -1e-6)


def test_maximum_diversification_favors_low_correlation_pair():
    # Two assets: equal vol, low correlation -> equal weights maximize diversification.
    cov = _cov_from_vols_corr(np.array([0.2, 0.2]), 0.0)
    w = maximum_diversification(cov)
    assert np.allclose(w, [0.5, 0.5], atol=1e-3)
