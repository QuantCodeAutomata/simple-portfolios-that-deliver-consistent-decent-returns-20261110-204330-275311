"""Vectorized backtest for the exhaustive and targeted equal-weight and
inverse-volatility ETF-subset search (paper Sec. 6.2).

The paper enumerates every 3-, 4- and 5-ETF combination of a filtered
universe and ranks each by Sharpe ratio, and separately by Sharpe ratio
followed by the lower-quartile yearly Sharpe ratio (Sec. 6.2). Running the
exact daily engine (``src.backtest.run_backtest``) once per combination is
far too slow for tens of thousands of combinations, so this module evaluates
the same monthly-rebalanced weighting rule -- equal weights, or weights
inverse to each asset's trailing volatility -- directly at monthly
frequency: one gross return per asset per inter-rebalance interval, the
target weights for a given combination, and the pre-trade weight drift
needed for the turnover-based trading cost. This is a deliberate simplified
approximation used only to rank tens of thousands of candidates; the
top-ranked combinations are re-evaluated with the exact daily engine
(``src.backtest.run_backtest``) in ``exp.exp1`` for the metrics finally
reported, matching the paper's own two-stage reporting (search then report).
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.backtest import monthly_rebalance_dates
from src.config import COV_LOOKBACK_L, TRADING_COST_KAPPA

WeightFn = Callable[[int], np.ndarray]


def monthly_interval_gross_returns(
    returns: pd.DataFrame, assets: list[str]
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Gross simple return of each asset over each (rebalance_{i-1}, rebalance_i] interval.

    Row 0 covers (returns.index[0], rebalance_dates[0]]; the panel's first
    daily return is NaN/0 by construction so the implied starting reference
    value at ``returns.index[0]`` is exactly 1.0.
    """
    reb = monthly_rebalance_dates(returns.index)
    sub = returns[assets].fillna(0.0)
    cum = (1.0 + sub).cumprod()
    cum_at_reb = cum.loc[reb]
    shifted = cum_at_reb.shift(1)
    shifted.iloc[0] = 1.0
    gross = (cum_at_reb / shifted).to_numpy()
    return gross, reb


def trailing_volatility(
    returns: pd.DataFrame, assets: list[str], reb_dates: pd.DatetimeIndex, lookback: int = COV_LOOKBACK_L,
) -> np.ndarray:
    """Trailing rolling-sample volatility sqrt((1/L) sum r^2) at each rebalance date.

    Matches ``src.risk_models.rolling_sample_covariance``'s diagonal, using
    only returns strictly before the rebalance date (no look-ahead), the
    same convention as ``src.portfolio_runner.make_strategy_fn``.
    """
    pos = {d: i for i, d in enumerate(returns.index)}
    reb_positions = np.array([pos[d] for d in reb_dates])
    r2 = (returns[assets].fillna(0.0).to_numpy()) ** 2
    prefix = np.vstack([np.zeros((1, r2.shape[1])), np.cumsum(r2, axis=0)])
    out = np.full((len(reb_positions), len(assets)), np.nan)
    for i, t in enumerate(reb_positions):
        lo = max(0, t - lookback)
        n = t - lo
        if n <= 0:
            continue
        out[i] = np.sqrt(np.maximum((prefix[t] - prefix[lo]) / n, 1e-12))
    return out


def simulate_static_weights(
    gross: np.ndarray, weight_fn: WeightFn, kappa: float = TRADING_COST_KAPPA,
) -> tuple[np.ndarray, np.ndarray]:
    """Monthly NAV and turnover path for a sequence of target weight vectors.

    At each month: a trading cost ``kappa * tau`` (turnover against the
    previous, drifted weights) is deducted from NAV, the target weights are
    then held fixed (no intra-month rebalancing) and grown by ``gross[i]``,
    matching the daily engine's "trade at the start, drift until the next
    rebalance" structure (Sec. 6.1) at monthly instead of daily resolution.
    """
    n_months, n_assets = gross.shape
    nav = np.empty(n_months)
    turnover = np.empty(n_months)
    v = 1.0
    w_pre = np.zeros(n_assets)
    for i in range(n_months):
        w_target = weight_fn(i)
        tau = 0.5 * np.abs(w_target - w_pre).sum()
        v *= (1.0 - kappa * tau)
        turnover[i] = tau
        denom = float(np.sum(w_target * gross[i]))
        v *= denom if denom > 0 else 1.0
        nav[i] = v
        w_pre = (w_target * gross[i]) / denom if denom > 0 else w_target
    return nav, turnover


def monthly_performance_summary(
    nav: np.ndarray, turnover: np.ndarray, reb_dates: pd.DatetimeIndex,
) -> dict[str, float]:
    """r, sigma, MDD, TO, SR, LQ (paper eq. 28) from a monthly NAV/turnover path.

    Annualizes with 12 monthly periods per year rather than 252 trading
    days, the monthly-frequency analogue of ``src.metrics.performance_summary``
    used only for the screening stage described in the module docstring.
    """
    nav_s = pd.Series(nav, index=reb_dates)
    prev = np.r_[1.0, nav[:-1]]
    monthly_returns = pd.Series(nav / prev - 1.0, index=reb_dates)
    t = len(monthly_returns)
    r_e = float((1.0 + monthly_returns).prod() ** (12.0 / t) - 1.0) if t else float("nan")
    sigma = float(monthly_returns.std(ddof=1) * np.sqrt(12.0)) if t > 1 else float("nan")
    running_max = nav_s.cummax()
    mdd = float(((running_max - nav_s) / running_max).max())
    sr = r_e / sigma if sigma and not np.isnan(sigma) else float("nan")
    turnover_s = pd.Series(turnover, index=reb_dates)
    to = float(turnover_s.groupby(turnover_s.index.year).sum().mean())

    years = monthly_returns.index.year
    yearly_sr = {}
    for y in sorted(set(years)):
        mask = years == y
        n_y = int(mask.sum())
        ry = float((1.0 + monthly_returns[mask]).prod() ** (12.0 / n_y) - 1.0)
        sy = float(monthly_returns[mask].std(ddof=1) * np.sqrt(12.0)) if n_y > 1 else float("nan")
        yearly_sr[y] = ry / sy if sy and not np.isnan(sy) else float("nan")
    lq = float(pd.Series(yearly_sr).quantile(0.25))
    return {"r": r_e, "sigma": sigma, "MDD": mdd, "TO": to, "SR": sr, "LQ": lq}


def _combo_iter(n_candidates: int, size: int, max_combos: int | None, rng_seed: int) -> Iterable[tuple[int, ...]]:
    all_combos = list(itertools.combinations(range(n_candidates), size))
    if max_combos is None or len(all_combos) <= max_combos:
        return all_combos
    rng = np.random.default_rng(rng_seed)
    chosen = rng.choice(len(all_combos), size=max_combos, replace=False)
    return [all_combos[i] for i in sorted(chosen)]


def search_combinations(
    returns: pd.DataFrame,
    candidate_assets: list[str],
    sizes: tuple[int, ...],
    method: str,
    kappa: float = TRADING_COST_KAPPA,
    cov_lookback: int = COV_LOOKBACK_L,
    max_combos: int | None = None,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Backtest every (or a random subsample of) combination of each size in ``sizes``.

    ``method`` is ``"equal"`` (Sec. 6.2 equal-weight exhaustive search) or
    ``"inverse_vol"`` (Sec. 6.2 inverse-volatility search). Returns one row
    per combination with its tickers, size, and the monthly-frequency
    performance summary.
    """
    assert method in ("equal", "inverse_vol"), method
    gross, reb = monthly_interval_gross_returns(returns, candidate_assets)
    inv_vol = None
    if method == "inverse_vol":
        vol = trailing_volatility(returns, candidate_assets, reb, cov_lookback)
        inv_vol = 1.0 / vol

    rows = []
    for size in sizes:
        for combo in _combo_iter(len(candidate_assets), size, max_combos, rng_seed):
            idx = list(combo)
            g = gross[:, idx]
            if method == "equal":
                w0 = np.full(size, 1.0 / size)
                nav, turnover = simulate_static_weights(g, lambda i, w0=w0: w0, kappa)
            else:
                iv = inv_vol[:, idx]

                def weight_fn(i: int, iv: np.ndarray = iv) -> np.ndarray:
                    row = iv[i]
                    return row / row.sum()

                nav, turnover = simulate_static_weights(g, weight_fn, kappa)
            summary = monthly_performance_summary(nav, turnover, reb)
            summary["tickers"] = tuple(candidate_assets[j] for j in idx)
            summary["size"] = size
            rows.append(summary)
    return pd.DataFrame(rows)


def rank_by_sr(results: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by Sharpe ratio only (Sec. 6.2, criterion (i))."""
    return results.sort_values("SR", ascending=False).reset_index(drop=True)


def rank_by_sr_then_lq(results: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates lexicographically by Sharpe ratio, then by LQ (Sec. 6.2, criterion (ii))."""
    return results.sort_values(["SR", "LQ"], ascending=False).reset_index(drop=True)
