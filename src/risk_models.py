"""Covariance forecasting methods (paper Sec. 4, Appendix A).

Three estimators share one interface, ``forecast(returns) -> (n, n) ndarray``,
operating on a trailing window of daily zero-mean returns (paper eq. before
Table 17 treats returns as zero mean, consistent with R^T R sample second
moments rather than demeaned covariances):

* ``RollingSampleCovariance``  -- Sec. 4.2, Sigma_hat = (1/L) R^T R.
* ``EwmaCovariance``           -- Sec. 4.2, separate half-lives for variances
  and correlations (Table 17).
* ``StatisticalFactorModel``   -- Sec. 4.3 / Appendix A, EM factor analysis
  with k statistical factors fitted on the EWMA-weighted correlation matrix
  of winsorized, volatility-standardized returns.

sklearn.decomposition.FactorAnalysis fits Lambda/Psi by EM but only from raw
equally-weighted samples, not from a precomputed EWMA-weighted correlation
matrix as Appendix A requires, so the EM recursion here is a custom
implementation operating directly on that matrix (classic Rubin-Thayer 1982
factor-analysis EM, as summarized in Appendix A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    COV_LOOKBACK_L,
    EWMA_HALFLIFE_VAR,
    EWMA_HALFLIFE_COV,
    FACTOR_EM_MAX_ITER,
    FACTOR_EM_TOL,
    N_STAT_FACTORS,
    WINSORIZE_Z,
)


def _halflife_to_lambda(halflife: float) -> float:
    """Decay factor lambda such that lambda**halflife == 0.5."""
    return float(np.exp(np.log(0.5) / halflife))


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Normalized EWMA weights for the n most recent observations (most recent last)."""
    lam = _halflife_to_lambda(halflife)
    age = np.arange(n - 1, -1, -1)  # most recent observation has age 0
    w = lam ** age
    return w / w.sum()


def rolling_sample_covariance(returns: np.ndarray, lookback: int = COV_LOOKBACK_L) -> np.ndarray:
    """Sigma_hat = (1/L) R^T R over the trailing ``lookback`` daily returns (Sec. 4.2)."""
    window = returns[-lookback:]
    L = window.shape[0]
    assert L > 0, "no returns available for the rolling covariance window"
    return (window.T @ window) / L


def ewma_covariance(
    returns: np.ndarray,
    halflife_var: float = EWMA_HALFLIFE_VAR,
    halflife_cov: float = EWMA_HALFLIFE_COV,
) -> np.ndarray:
    """EWMA covariance with separate half-lives for variances and correlations (Sec. 4.2, Table 17).

    Variances are forecast with half-life H_var, correlations with the longer
    half-life H_cov, and the two are recombined as Sigma = D Corr D.
    """
    n_obs = returns.shape[0]
    w_var = _ewma_weights(n_obs, halflife_var)
    w_cov = _ewma_weights(n_obs, halflife_cov)

    var = w_var @ (returns ** 2)
    var = np.maximum(var, 1e-12)
    vol = np.sqrt(var)

    standardized = returns / vol
    corr = (standardized * w_cov[:, None]).T @ standardized
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    sigma = np.outer(vol, vol) * corr
    return sigma


@dataclass
class FactorModelResult:
    loadings: np.ndarray       # F, (n_assets, k)
    idio_var: np.ndarray       # diagonal of D, (n_assets,), in correlation units
    vol: np.ndarray            # per-asset EWMA volatility used to rescale back to covariance
    n_iter: int

    def covariance(self) -> np.ndarray:
        """Sigma = diag(vol) (F F^T + D) diag(vol) (Sec. 4.3)."""
        corr = self.loadings @ self.loadings.T + np.diag(self.idio_var)
        return np.outer(self.vol, self.vol) * corr


def _factor_analysis_em(
    corr: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """EM for the factor-analysis model corr ~= F F^T + diag(psi) (Appendix A).

    Initializes F from the leading k eigenpairs of ``corr`` and iterates the
    Rubin-Thayer (1982) EM recursion until the Frobenius norm of the fitted
    matrix changes by less than ``tol`` or ``max_iter`` is reached.
    """
    n = corr.shape[0]
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order][:k], eigvecs[:, order][:, :k]
    eigvals = np.maximum(eigvals, 1e-8)
    F = eigvecs * np.sqrt(eigvals)[None, :]
    psi = np.maximum(np.diag(corr) - np.sum(F ** 2, axis=1), 1e-6)

    prev_fit = F @ F.T + np.diag(psi)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        psi_inv = 1.0 / psi
        Fp = F * psi_inv[:, None]
        M = np.eye(k) + F.T @ Fp               # I + F^T Psi^-1 F, (k, k)
        M_inv = np.linalg.inv(M)
        beta = M_inv @ Fp.T                      # (k, n): E[factor | data]-style gain

        delta = np.eye(k) - beta @ F + beta @ corr @ beta.T  # posterior factor covariance
        F_new = corr @ beta.T @ np.linalg.inv(delta)
        psi_new = np.maximum(np.diag(corr) - np.sum((F_new @ beta) * corr, axis=1), 1e-6)

        fit = F_new @ F_new.T + np.diag(psi_new)
        change = np.linalg.norm(fit - prev_fit) / max(np.linalg.norm(prev_fit), 1e-12)
        F, psi, prev_fit = F_new, psi_new, fit
        if change < tol:
            break
    return F, psi, n_iter


def fit_statistical_factor_model(
    returns: np.ndarray,
    k: int = N_STAT_FACTORS,
    halflife_var: float = EWMA_HALFLIFE_VAR,
    halflife_cov: float = EWMA_HALFLIFE_COV,
    winsorize_z: float = WINSORIZE_Z,
    max_iter: int = FACTOR_EM_MAX_ITER,
    tol: float = FACTOR_EM_TOL,
) -> FactorModelResult:
    """Fit the k-factor statistical risk model (Sec. 4.3, Appendix A).

    Steps: standardize returns by their EWMA volatility, winsorize the
    standardized returns at +/- ``winsorize_z``, form the EWMA-weighted
    correlation matrix, initialize F from its leading k eigenpairs, and run
    EM until convergence. The fitted correlation is rescaled back to a
    covariance using the same EWMA volatilities.
    """
    n_obs = returns.shape[0]
    w_var = _ewma_weights(n_obs, halflife_var)
    var = np.maximum(w_var @ (returns ** 2), 1e-12)
    vol = np.sqrt(var)

    standardized = np.clip(returns / vol, -winsorize_z, winsorize_z)

    w_cov = _ewma_weights(n_obs, halflife_cov)
    corr = (standardized * w_cov[:, None]).T @ standardized
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    F, psi, n_iter = _factor_analysis_em(corr, k=k, max_iter=max_iter, tol=tol)
    return FactorModelResult(loadings=F, idio_var=psi, vol=vol, n_iter=n_iter)


def explained_variance_by_pc(returns: np.ndarray) -> np.ndarray:
    """Cumulative share of variance explained by the leading principal components.

    Used to motivate k = 10 (Sec. 6.3): the first PC explains about half of all
    variance and the first ten about 89%.
    """
    corr = np.corrcoef(returns, rowvar=False)
    eigvals = np.linalg.eigvalsh(corr)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    return np.cumsum(eigvals) / np.sum(eigvals)
