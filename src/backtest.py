"""Daily backtest engine (paper Sec. 6.1).

Holdings h_t and cash c_t evolve daily using realized asset returns r_t and
the risk-free cash return r_f,t. NAV v_t = 1^T h_t + c_t and weights
w_t = h_t / v_t. On non-trading days, only holdings and cash drift. On
monthly trading dates the strategy's target weights are computed, the
portfolio trades at the start of day t using the previous close, turnover
tau_t = 0.5 * ||w_t - w_pre||_1 is charged at rate kappa against cash, and
holdings are reset to the target weights.

A strategy is any callable ``(date, history) -> np.ndarray`` returning target
weights over the tradable universe at that date (history is a dict of
pre-computed per-asset return panels truncated to information available up to
and including the previous close, preventing look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.config import MIN_HISTORY_DAYS, TRADING_COST_KAPPA


StrategyFn = Callable[[pd.Timestamp, int], np.ndarray]


@dataclass
class BacktestResult:
    nav: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series             # per-day turnover (zero on non-trading days)
    portfolio_returns: pd.Series    # daily NAV return, net of costs
    excess_returns: pd.Series       # portfolio_returns - risk-free daily return
    rebalance_dates: pd.DatetimeIndex


def monthly_rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month within ``dates`` (Sec. 6.1).

    The paper specifies monthly trading but not the exact day-of-month
    convention (recorded as an open point in results/RESULTS.md); this
    implementation rebalances on the last available trading day of each
    month, the standard convention for monthly-rebalanced equity/ETF
    backtests.
    """
    s = pd.Series(dates, index=dates)
    last_of_month = s.groupby([dates.year, dates.month]).max()
    return pd.DatetimeIndex(sorted(last_of_month.values))


def is_investable(listing_day_index: dict[str, int], ticker: str, t_index: int, min_history: int = MIN_HISTORY_DAYS) -> bool:
    """True once ``ticker`` has at least ``min_history`` trading days of history (Sec. 6.1)."""
    first = listing_day_index.get(ticker)
    if first is None:
        return False
    return (t_index - first) >= min_history


def run_backtest(
    returns: pd.DataFrame,
    strategy_fn: StrategyFn,
    risk_free_daily: pd.Series,
    kappa: float = TRADING_COST_KAPPA,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> BacktestResult:
    """Run the daily engine described in Sec. 6.1.

    Parameters
    ----------
    returns : DataFrame (dates x tickers) of daily simple total returns;
        NaN before an asset is investable.
    strategy_fn : callable(date, t_index) -> dict[ticker, weight], the target
        weights on tickers investable at that date; no cash key (cash is the
        residual 1 - sum(weights)).
    risk_free_daily : Series of daily cash returns aligned to ``returns.index``.
    kappa : half-spread trading cost rate per dollar traded (turnover).
    rebalance_dates : trading dates on which the strategy is invoked; defaults
        to the last trading day of each month (``monthly_rebalance_dates``).
    """
    dates = returns.index
    tickers = list(returns.columns)
    n = len(tickers)
    if rebalance_dates is None:
        rebalance_dates = monthly_rebalance_dates(dates)
    rebalance_set = set(rebalance_dates)

    h = np.zeros(n)          # dollar holdings per asset
    c = 1.0                  # cash, NAV normalized to start at 1
    nav_path = np.zeros(len(dates))
    weight_path = np.zeros((len(dates), n))
    turnover_path = np.zeros(len(dates))
    ret_path = np.zeros(len(dates))

    r_mat = returns.fillna(0.0).to_numpy()
    rf = risk_free_daily.reindex(dates).fillna(0.0).to_numpy()

    for t, date in enumerate(dates):
        if t > 0:
            h = h * (1.0 + r_mat[t])
            c = c * (1.0 + rf[t])
        v = h.sum() + c
        assert v > 0, f"NAV went non-positive at {date}"

        if date in rebalance_set:
            w_pre = h / v
            w_target_dict = strategy_fn(date, t)
            w_target = np.zeros(n)
            for i, tkr in enumerate(tickers):
                w_target[i] = w_target_dict.get(tkr, 0.0)
            assert np.all(w_target >= -1e-9), "no shorting allowed"
            assert w_target.sum() <= 1.0 + 1e-6, "no leverage allowed"

            tau = 0.5 * np.abs(w_target - w_pre).sum()
            cost = kappa * tau * v
            v_after_cost = v - cost
            h = w_target * v_after_cost
            c = v_after_cost - h.sum()
            turnover_path[t] = tau
            v = v_after_cost

        nav_path[t] = v
        weight_path[t] = h / v
        ret_path[t] = (v / nav_path[t - 1] - 1.0) if t > 0 else 0.0

    nav = pd.Series(nav_path, index=dates, name="nav")
    weights = pd.DataFrame(weight_path, index=dates, columns=tickers)
    turnover = pd.Series(turnover_path, index=dates, name="turnover")
    portfolio_returns = pd.Series(ret_path, index=dates, name="portfolio_return")
    excess_returns = (portfolio_returns - rf).rename("excess_return")

    return BacktestResult(
        nav=nav,
        weights=weights,
        turnover=turnover,
        portfolio_returns=portfolio_returns,
        excess_returns=excess_returns,
        rebalance_dates=rebalance_dates,
    )
