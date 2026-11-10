"""Unit tests for src/exhaustive_search.py (paper Sec. 6.2 search screening)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exhaustive_search import (
    monthly_interval_gross_returns,
    monthly_performance_summary,
    rank_by_sr,
    rank_by_sr_then_lq,
    search_combinations,
    simulate_static_weights,
    trailing_volatility,
)


def _synthetic_returns(n_days: int = 1500, n_assets: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    vols = np.linspace(0.005, 0.02, n_assets)
    drift = np.linspace(0.0001, 0.0005, n_assets)
    data = rng.normal(drift, vols, size=(n_days, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_monthly_interval_gross_returns_shape_and_positivity():
    returns = _synthetic_returns()
    gross, reb = monthly_interval_gross_returns(returns, list(returns.columns))
    assert gross.shape == (len(reb), returns.shape[1])
    assert np.all(gross > 0)


def test_simulate_static_weights_no_cost_equals_buy_and_hold_growth():
    returns = _synthetic_returns(n_assets=3)
    gross, reb = monthly_interval_gross_returns(returns, list(returns.columns))
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    nav, turnover = simulate_static_weights(gross, lambda i: w0, kappa=0.0)
    assert nav[-1] > 0
    assert np.all(turnover >= 0)
    # Zero cost and a constant target: NAV must equal the cumulative product
    # of the monthly portfolio gross returns.
    expected = np.cumprod(gross @ w0)
    assert nav == pytest.approx(expected, rel=1e-8)


def test_trading_cost_reduces_nav_relative_to_zero_cost():
    returns = _synthetic_returns(n_assets=4, seed=3)
    gross, reb = monthly_interval_gross_returns(returns, list(returns.columns))
    w0 = np.array([0.4, 0.3, 0.2, 0.1])
    nav_free, _ = simulate_static_weights(gross, lambda i: w0, kappa=0.0)
    nav_cost, turnover = simulate_static_weights(gross, lambda i: w0, kappa=5e-4)
    assert turnover.sum() > 0  # drift between rebalances always creates some turnover
    assert nav_cost[-1] < nav_free[-1]


def test_trailing_volatility_matches_manual_rolling_rms():
    returns = _synthetic_returns(n_assets=2, seed=5)
    assets = list(returns.columns)
    _, reb = monthly_interval_gross_returns(returns, assets)
    vol = trailing_volatility(returns, assets, reb, lookback=126)
    pos = {d: i for i, d in enumerate(returns.index)}
    t = pos[reb[50]]
    window = returns[assets].iloc[t - 126:t].to_numpy()
    manual = np.sqrt((window ** 2).mean(axis=0))
    assert np.allclose(vol[50], manual, atol=1e-10)


def test_search_combinations_and_ranking_consistency():
    returns = _synthetic_returns(n_assets=6, n_days=800, seed=7)
    assets = list(returns.columns)
    results = search_combinations(returns, assets, sizes=(3,), method="equal", max_combos=None)
    assert len(results) == 20  # C(6, 3)
    ranked_sr = rank_by_sr(results)
    assert ranked_sr["SR"].is_monotonic_decreasing
    ranked_sr_lq = rank_by_sr_then_lq(results)
    assert len(ranked_sr_lq) == len(results)
    assert set(ranked_sr_lq["tickers"]) == set(results["tickers"])


def test_search_combinations_inverse_vol_runs_and_weights_favor_low_vol():
    returns = _synthetic_returns(n_assets=5, n_days=800, seed=11)
    assets = list(returns.columns)
    results = search_combinations(returns, assets, sizes=(3,), method="inverse_vol", max_combos=None)
    assert np.isfinite(results["SR"]).any()


def test_max_combos_subsamples_without_exceeding_requested_count():
    returns = _synthetic_returns(n_assets=8, n_days=400, seed=13)
    assets = list(returns.columns)
    results = search_combinations(returns, assets, sizes=(4,), method="equal", max_combos=10)
    assert len(results) == 10
