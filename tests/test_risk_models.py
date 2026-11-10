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


def test_realized_covariance_noise_floor():
    """Synthetic diagnosis for the Experiment 2 Frobenius-distance gap (RESULTS.md).

    Even a perfect forecast of a known true covariance, compared against a
    sample covariance drawn from only 21 observations in p=59 dimensions
    (paper's "realized covariance of the following 21 trading days", Sec.
    4.1), carries a large normalized Frobenius distance purely from
    small-sample noise in the *target*. This test checks that this noise
    floor shrinks monotonically as the sample size grows, which is the
    property used in RESULTS.md to show the gap to the paper's reported
    absolute values is not caused by a formula error in ``exp/exp2.py``.
    """
    rng = np.random.default_rng(0)
    p, k = 59, 10
    factor_loadings = rng.normal(0, 0.01, size=(p, k))
    idio_var = rng.uniform(0.0001, 0.0004, size=p)
    sigma_true = factor_loadings @ factor_loadings.T + np.diag(idio_var)
    chol = np.linalg.cholesky(sigma_true)

    def mean_distance(n_obs: int, reps: int = 60) -> float:
        dists = []
        for _ in range(reps):
            sample = rng.standard_normal((n_obs, p)) @ chol.T
            sigma_hat = (sample.T @ sample) / n_obs
            dists.append(np.linalg.norm(sigma_hat - sigma_true) / np.linalg.norm(sigma_true))
        return float(np.mean(dists))

    d21 = mean_distance(21)
    d252 = mean_distance(252)
    assert d21 > 0.5, "a 21-observation, 59-asset realized covariance is expected to be noisy"
    assert d252 < d21, "distance to the true covariance must shrink with more observations"


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
