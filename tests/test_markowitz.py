"""Unit tests for src/markowitz.py (paper Sec. 3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.markowitz import solve_markowitz


def _diag_cov(vols: np.ndarray) -> np.ndarray:
    return np.diag(vols ** 2)


def test_markowitz_respects_box_and_sum_constraints():
    n = 5
    alpha = np.array([0.10, 0.05, 0.02, -0.01, 0.0])
    w_pre = np.zeros(n)
    cov = _diag_cov(np.full(n, 0.15))
    w = solve_markowitz(alpha, w_pre, cov=cov, w_max=0.10, turnover_limit=0.10, delta_w=0.05)
    assert np.all(w >= -1e-6)
    assert np.all(w <= 0.10 + 1e-6)
    assert w.sum() <= 1.0 + 1e-6


def test_markowitz_turnover_cap_binds():
    # Only asset 0 earns alpha, so an unconstrained solver would push it to
    # w_max; with a turnover cap tighter than w_max and delta_w, the L1 trade
    # distance ||w - w_pre||_1 <= T_tgt (Sec. 3.3) is the binding constraint.
    n = 3
    alpha = np.array([1.0, 0.0, 0.0])
    w_pre = np.zeros(n)
    cov = _diag_cov(np.full(n, 0.10))
    w = solve_markowitz(alpha, w_pre, cov=cov, turnover_limit=0.03, delta_w=0.10, w_max=0.10)
    l1_trade = np.abs(w - w_pre).sum()
    assert l1_trade <= 0.03 + 1e-6
    assert w[0] == pytest.approx(0.03, abs=1e-4)


def test_markowitz_per_asset_delta_cap():
    n = 2
    alpha = np.array([1.0, 0.0])
    w_pre = np.array([0.0, 0.0])
    cov = _diag_cov(np.full(n, 0.05))
    w = solve_markowitz(alpha, w_pre, cov=cov, turnover_limit=1.0, delta_w=0.03, w_max=0.10)
    assert np.all(np.abs(w - w_pre) <= 0.03 + 1e-6)


def test_markowitz_risk_constraint_binds_with_high_alpha():
    n = 4
    alpha = np.ones(n)
    w_pre = np.zeros(n)
    cov = _diag_cov(np.full(n, 0.50))  # high vol forces the risk budget to bind
    w = solve_markowitz(alpha, w_pre, cov=cov, sigma_tgt=0.08, turnover_limit=1.0, delta_w=1.0, w_max=1.0)
    realized_vol = np.sqrt(w @ cov @ w)
    assert realized_vol <= 0.08 + 1e-4


def test_markowitz_factor_form_matches_dense_covariance_form():
    n = 4
    rng = np.random.default_rng(0)
    F = rng.normal(0, 0.1, (n, 2))
    d = np.full(n, 0.01)
    cov = F @ F.T + np.diag(d)
    alpha = np.array([0.05, 0.03, -0.02, 0.01])
    w_pre = np.zeros(n)
    w_dense = solve_markowitz(alpha, w_pre, cov=cov, sigma_tgt=0.08)
    w_factor = solve_markowitz(alpha, w_pre, factor_loadings=F, idio_var=d, sigma_tgt=0.08)
    assert np.allclose(w_dense, w_factor, atol=1e-3)


def test_markowitz_requires_a_risk_specification():
    with pytest.raises(AssertionError):
        solve_markowitz(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
