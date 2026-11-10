"""Unit tests for src/backtest.py (paper Sec. 6.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import is_investable, monthly_rebalance_dates, run_backtest


def test_monthly_rebalance_dates_last_trading_day_per_month():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    reb = monthly_rebalance_dates(dates)
    assert len(reb) == 3
    assert reb[0] == pd.Timestamp("2020-01-31")
    assert reb[-1] == pd.Timestamp("2020-03-31")


def test_is_investable_requires_minimum_history():
    listing = {"AAA": 10}
    assert not is_investable(listing, "AAA", t_index=100, min_history=126)
    assert is_investable(listing, "AAA", t_index=136, min_history=126)
    assert not is_investable(listing, "BBB", t_index=200, min_history=126)


def test_backtest_no_shorting_no_leverage_and_nav_positive():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, (len(dates), 2)), index=dates, columns=["A", "B"]
    )
    rf = pd.Series(0.00005, index=dates)

    def strategy(date, t_index):
        return {"A": 0.5, "B": 0.5}

    result = run_backtest(returns, strategy, rf)
    assert (result.weights.to_numpy() >= -1e-9).all()
    assert (result.weights.sum(axis=1) <= 1.0 + 1e-6).all()
    assert (result.nav > 0).all()
    # Monthly rebalancing produces turnover only on rebalance dates.
    non_reb = ~result.turnover.index.isin(result.rebalance_dates)
    assert np.allclose(result.turnover[non_reb], 0.0)


def test_backtest_charges_cost_proportional_to_turnover():
    dates = pd.bdate_range("2020-01-01", "2020-02-29")
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    rf = pd.Series(0.0, index=dates)

    def strategy(date, t_index):
        # Flip fully between A and B each rebalance -> turnover = 1 each time.
        first_reb = monthly_rebalance_dates(dates)[0]
        return {"A": 1.0, "B": 0.0} if date == first_reb else {"A": 0.0, "B": 1.0}

    kappa = 5e-4
    result = run_backtest(returns, strategy, rf, kappa=kappa)
    reb_dates = result.rebalance_dates
    # Second rebalance flips 100% of the (then) portfolio -> turnover ~= 1.
    assert result.turnover[reb_dates[1]] == pytest.approx(1.0, abs=1e-6)
    # NAV must drop by roughly kappa * turnover at that date, net of zero returns.
    assert result.nav[reb_dates[1]] < result.nav[reb_dates[0]]


def test_backtest_no_lookahead_strategy_only_sees_past_index():
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 2)), index=dates, columns=["A", "B"]
    )
    rf = pd.Series(0.0, index=dates)
    seen_indices = []

    def strategy(date, t_index):
        seen_indices.append(t_index)
        assert dates[t_index] == date
        return {"A": 0.5, "B": 0.5}

    run_backtest(returns, strategy, rf)
    # The strategy must be invoked using the index as-of the current date only
    # (never an index beyond the current position in the date array).
    assert all(idx < len(dates) for idx in seen_indices)
