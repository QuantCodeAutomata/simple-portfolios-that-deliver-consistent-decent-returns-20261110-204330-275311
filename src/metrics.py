"""Common performance metrics (paper Sec. 6.1, eq. 28).

All functions take a daily portfolio-return series net of the risk-free rate
(excess returns) plus the daily turnover series produced by the backtest
engine in backtest.py, and report the metrics shared by every experiment:
annualized geometric excess return r_e, annualized volatility sigma, maximum
drawdown MDD, Sharpe ratio SR, average annual turnover TO, and the lower
quartile LQ of calendar-year Sharpe ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANNUALIZATION_DAYS


def annualized_geometric_return(excess_returns: pd.Series) -> float:
    """Annualized geometric mean of daily excess returns (paper eq. 28, r_e).

    r_e = (prod_t (1 + r_t))^(252/T) - 1, the realized compound excess return
    annualized geometrically, as opposed to the arithmetic mean which
    overstates compound performance when returns vary (Sec. 6.1).
    """
    t = len(excess_returns)
    if t == 0:
        return float("nan")
    gross = (1.0 + excess_returns).prod()
    return float(gross ** (ANNUALIZATION_DAYS / t) - 1.0)


def annualized_volatility(portfolio_returns: pd.Series) -> float:
    """Annualized realized volatility of the portfolio return itself (eq. 28, sigma)."""
    if len(portfolio_returns) < 2:
        return float("nan")
    return float(portfolio_returns.std(ddof=1) * np.sqrt(ANNUALIZATION_DAYS))


def max_drawdown(nav: pd.Series) -> float:
    """Maximum running drawdown of the NAV series (eq. 28, MDD).

    D_t = max_{tau<=t} (V_tau - V_t) / V_tau; MDD = max_t D_t.
    """
    running_max = nav.cummax()
    drawdown = (running_max - nav) / running_max
    return float(drawdown.max())


def sharpe_ratio(r_e: float, sigma: float) -> float:
    """SR = r_e / sigma (eq. 28)."""
    if sigma == 0 or np.isnan(sigma):
        return float("nan")
    return float(r_e / sigma)


def yearly_sharpe_ratios(excess_returns: pd.Series, portfolio_returns: pd.Series) -> pd.Series:
    """Annualized Sharpe ratio within each calendar year (eq. 28, SR_y)."""
    years = excess_returns.index.year
    out = {}
    for year in sorted(set(years)):
        mask = years == year
        re_y = annualized_geometric_return(excess_returns[mask])
        sig_y = annualized_volatility(portfolio_returns[mask])
        out[year] = sharpe_ratio(re_y, sig_y)
    return pd.Series(out)


def lower_quartile_sharpe(excess_returns: pd.Series, portfolio_returns: pd.Series) -> float:
    """LQ, the empirical 25th percentile of calendar-year Sharpe ratios (eq. 28)."""
    sr_y = yearly_sharpe_ratios(excess_returns, portfolio_returns)
    if sr_y.empty:
        return float("nan")
    return float(sr_y.quantile(0.25))


def average_annual_turnover(turnover: pd.Series) -> float:
    """TO, the average over calendar years of the sum of per-rebalance turnover (Sec. 6.1).

    A TO of 0.20 means a fifth of the portfolio changed in a typical year.
    """
    if turnover.empty:
        return 0.0
    yearly_sum = turnover.groupby(turnover.index.year).sum()
    return float(yearly_sum.mean())


def performance_summary(
    nav: pd.Series,
    portfolio_returns: pd.Series,
    excess_returns: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    """Compute the full metric set (r, sigma, MDD, TO, SR, LQ) used in every table."""
    r_e = annualized_geometric_return(excess_returns)
    sigma = annualized_volatility(portfolio_returns)
    mdd = max_drawdown(nav)
    to = average_annual_turnover(turnover)
    sr = sharpe_ratio(r_e, sigma)
    lq = lower_quartile_sharpe(excess_returns, portfolio_returns)
    return {"r": r_e, "sigma": sigma, "MDD": mdd, "TO": to, "SR": sr, "LQ": lq}
