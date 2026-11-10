"""Unit tests for src/metrics.py (paper Sec. 6.1, eq. 28)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_geometric_return,
    annualized_volatility,
    average_annual_turnover,
    lower_quartile_sharpe,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    yearly_sharpe_ratios,
)


def _daily_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def test_annualized_geometric_return_constant_rate():
    idx = _daily_index(252)
    daily = (1.10) ** (1 / 252) - 1.0
    r = pd.Series(daily, index=idx)
    assert annualized_geometric_return(r) == pytest.approx(0.10, abs=1e-6)


def test_annualized_geometric_return_empty():
    assert np.isnan(annualized_geometric_return(pd.Series(dtype=float)))


def test_annualized_volatility_scales_with_sqrt_252():
    idx = _daily_index(252)
    rng = np.random.default_rng(0)
    daily_sigma = 0.01
    r = pd.Series(rng.normal(0, daily_sigma, len(idx)), index=idx)
    vol = annualized_volatility(r)
    assert vol == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert vol > 0


def test_max_drawdown_known_path():
    nav = pd.Series([1.0, 1.2, 0.9, 1.1], index=_daily_index(4))
    # Peak 1.2 -> trough 0.9: drawdown = (1.2 - 0.9) / 1.2 = 0.25
    assert max_drawdown(nav) == pytest.approx(0.25)


def test_max_drawdown_monotonic_increase_is_zero():
    nav = pd.Series([1.0, 1.1, 1.2, 1.3], index=_daily_index(4))
    assert max_drawdown(nav) == pytest.approx(0.0)


def test_sharpe_ratio_sign_and_zero_vol():
    assert sharpe_ratio(0.1, 0.2) == pytest.approx(0.5)
    assert sharpe_ratio(-0.1, 0.2) < 0
    assert np.isnan(sharpe_ratio(0.1, 0.0))


def test_yearly_sharpe_and_lower_quartile_length():
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    rng = np.random.default_rng(1)
    excess = pd.Series(rng.normal(0.0003, 0.01, len(idx)), index=idx)
    ret = excess + 0.0001
    sr_y = yearly_sharpe_ratios(excess, ret)
    assert set(sr_y.index) == {2018, 2019, 2020, 2021}
    lq = lower_quartile_sharpe(excess, ret)
    assert lq <= sr_y.median()


def test_average_annual_turnover_sums_within_year():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    turnover = pd.Series(0.0, index=idx)
    turnover.iloc[::21] = 0.05  # roughly monthly rebalances
    to = average_annual_turnover(turnover)
    yearly = turnover.groupby(turnover.index.year).sum()
    assert to == pytest.approx(yearly.mean())


def test_performance_summary_keys():
    idx = _daily_index(252)
    rng = np.random.default_rng(2)
    ret = pd.Series(rng.normal(0.0003, 0.008, len(idx)), index=idx)
    nav = (1 + ret).cumprod()
    turnover = pd.Series(0.0, index=idx)
    summary = performance_summary(nav, ret, ret, turnover)
    assert set(summary) == {"r", "sigma", "MDD", "TO", "SR", "LQ"}
    assert 0.0 <= summary["MDD"] <= 1.0
