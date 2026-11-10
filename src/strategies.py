"""Simple portfolio construction strategies (paper Sec. 3.2).

Each strategy maps a covariance forecast (and, for fixed weights, a
predetermined allocation) to a weight vector summing to at most one, with the
residual held in cash. All six variants described in Sec. 3.2 are implemented:
fixed weights, volatility control, inverse volatility, minimum volatility,
risk parity, and maximum diversification. Minimum volatility, risk parity and
maximum diversification are solved with CVXPY, following the paper's own
CVXPY-style formulations (Sec. 3.2, Appendix C code listings).

Context7 (cvxpy) confirmed `quad_form` and the exponential/log atoms used
below; PyPortfolioOpt and Riskfolio-Lib were not used because the paper gives
its own custom convex formulations (risk parity with an explicit cash slack,
maximum diversification in correlation space) that differ from those
packages' default objectives.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.config import RISK_PARITY_RHO, TARGET_VOL

_SOLVER_KWARGS = {"solver": cp.CLARABEL}


def fixed_weights(weights: np.ndarray) -> np.ndarray:
    """Predetermined weights, used directly (Sec. 3.2 (a))."""
    w = np.asarray(weights, dtype=float)
    assert np.all(w >= -1e-9), "fixed weights must be non-negative (no shorting)"
    assert w.sum() <= 1.0 + 1e-6, "fixed weights must not exceed full investment (no leverage)"
    return w


def volatility_control(weights: np.ndarray, cov: np.ndarray, sigma_tgt: float = TARGET_VOL) -> np.ndarray:
    """Scale ``weights`` so realized risk targets ``sigma_tgt``, capped at 1 (Sec. 3.2 (b)).

    theta = min(1, sigma_tgt / sqrt(w^T Sigma w)); the remainder (1 - theta)
    of NAV is held in cash, i.e. the returned vector may sum to less than one.
    """
    w = np.asarray(weights, dtype=float)
    port_vol = np.sqrt(max(w @ cov @ w, 1e-16)) * np.sqrt(252)
    theta = min(1.0, sigma_tgt / port_vol) if port_vol > 0 else 1.0
    return theta * w


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    """w_i proportional to sigma_i^{-1}, normalized to sum to one (Sec. 3.2 (c))."""
    vol = np.sqrt(np.diag(cov))
    inv = 1.0 / np.maximum(vol, 1e-12)
    return inv / inv.sum()


def minimum_volatility(cov: np.ndarray) -> np.ndarray:
    """min w^T Sigma w s.t. 1^T w = 1, w >= 0 (Sec. 3.2 (d))."""
    n = cov.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1, w >= 0])
    prob.solve(**_SOLVER_KWARGS)
    assert prob.status in ("optimal", "optimal_inaccurate"), f"min-vol solve failed: {prob.status}"
    return np.clip(np.asarray(w.value).flatten(), 0, None)


def risk_parity(cov: np.ndarray, rho: float = RISK_PARITY_RHO) -> np.ndarray:
    """min 0.5 w^T Sigma w - rho * 1^T log(w) s.t. 1^T w + c = 1, w > 0, c >= 0 (Sec. 3.2 (e)).

    The log-barrier equalizes marginal risk contributions across assets while
    the explicit cash slack c lets the solution hold less than full NAV.
    """
    n = cov.shape[0]
    w = cp.Variable(n)
    c = cp.Variable(nonneg=True)
    objective = 0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - rho * cp.sum(cp.log(w))
    prob = cp.Problem(cp.Minimize(objective), [cp.sum(w) + c == 1, w >= 1e-8])
    prob.solve(**_SOLVER_KWARGS)
    assert prob.status in ("optimal", "optimal_inaccurate"), f"risk-parity solve failed: {prob.status}"
    return np.clip(np.asarray(w.value).flatten(), 0, None)


def maximum_diversification(cov: np.ndarray) -> np.ndarray:
    """Correlation-space QP, then rescale back to weights (Sec. 3.2 (f)).

    min y^T C y s.t. 1^T y = 1, y >= 0, where C is the correlation matrix; the
    maximum-diversification weights are recovered as
    w = D_sigma^{-1} y / (1^T D_sigma^{-1} y), D_sigma = diag(vol).
    """
    vol = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / np.maximum(vol, 1e-12)
    corr = cov * np.outer(inv_vol, inv_vol)
    np.fill_diagonal(corr, 1.0)

    n = cov.shape[0]
    y = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(corr))), [cp.sum(y) == 1, y >= 0])
    prob.solve(**_SOLVER_KWARGS)
    assert prob.status in ("optimal", "optimal_inaccurate"), f"max-diversification solve failed: {prob.status}"
    y_val = np.clip(np.asarray(y.value).flatten(), 0, None)

    w_unnorm = inv_vol * y_val
    return w_unnorm / w_unnorm.sum()
