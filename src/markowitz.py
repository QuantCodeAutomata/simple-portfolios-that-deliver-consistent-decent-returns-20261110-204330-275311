"""Modified long-only Markowitz allocation (paper Sec. 3.3).

    maximize   alpha^T w - kappa * ||w - w_pre||_1
    subject to 1^T w + c = 1,  c >= 0,  w >= 0
               w^T Sigma w <= sigma_tgt^2
               ||w - w_pre||_1 <= T_tgt
               |w - w_pre| <= delta_w
               0 <= w <= w_max

Context7 (cvxpy) confirmed the second-order-cone formulation of a quadratic
risk constraint via ``cp.SOC``/``cp.norm2``; when a factor decomposition
Sigma = F F^T + diag(d) is available the risk constraint is imposed as
``cp.norm2(vstack([F.T @ w, sqrt(d) * w])) <= sigma_tgt``, exactly the
paper's Appendix C code listing, avoiding a dense quadratic constraint.
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from src.config import (
    MARKOWITZ_DELTA_W,
    MARKOWITZ_TURNOVER_LIMIT,
    MARKOWITZ_W_MAX,
    TARGET_VOL,
    TRADING_COST_KAPPA,
)

_SOLVER_KWARGS = {"solver": cp.CLARABEL}


def solve_markowitz(
    alpha: np.ndarray,
    w_pre: np.ndarray,
    cov: np.ndarray | None = None,
    factor_loadings: np.ndarray | None = None,
    idio_var: np.ndarray | None = None,
    sigma_tgt: float = TARGET_VOL,
    turnover_limit: float = MARKOWITZ_TURNOVER_LIMIT,
    delta_w: float = MARKOWITZ_DELTA_W,
    w_max: float = MARKOWITZ_W_MAX,
    kappa: float = TRADING_COST_KAPPA,
) -> np.ndarray:
    """Solve the modified Markowitz problem of Sec. 3.3.

    Either ``cov`` (dense covariance, annualized) or both
    ``factor_loadings`` (F) and ``idio_var`` (diagonal of D) must be given;
    the factor form reproduces the paper's Appendix C SOC implementation.
    """
    n = len(alpha)
    w = cp.Variable(n)
    trade = w - w_pre

    objective = alpha @ w - kappa * cp.norm1(trade)
    constraints = [
        cp.sum(w) <= 1.0,
        w >= 0,
        cp.norm1(trade) <= turnover_limit,
        cp.abs(trade) <= delta_w,
        w <= w_max,
    ]
    if factor_loadings is not None and idio_var is not None:
        d_half = np.sqrt(np.maximum(idio_var, 1e-12))
        risk_vec = cp.hstack([factor_loadings.T @ w, cp.multiply(d_half, w)])
        constraints.append(cp.norm2(risk_vec) <= sigma_tgt)
    else:
        assert cov is not None, "either cov or (factor_loadings, idio_var) must be provided"
        constraints.append(cp.quad_form(w, cp.psd_wrap(cov)) <= sigma_tgt ** 2)

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(**_SOLVER_KWARGS)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Infeasible only if w_pre itself violates a hard cap (should not
        # happen because every prior solution respects the same caps); a
        # pathological numerical failure falls back to holding w_pre.
        return np.clip(w_pre, 0, w_max)
    return np.clip(np.asarray(w.value).flatten(), 0, None)
