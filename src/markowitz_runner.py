"""Wiring between alpha signals, the statistical factor risk model (Sec. 4.3)
and the modified long-only Markowitz allocator (Sec. 3.3) into one
``StrategyFn`` consumed by the daily backtest engine (``src.backtest``).

Shared by the full-universe / category-ablation alpha-model experiment
(``exp.exp3``) and the final proposed "Momentum Universe" and regret-bound
portfolios (``exp.exp4``), so both reuse one implementation of "how the alpha
vector and the risk model are combined into a Markowitz trade" rather than
each inventing its own.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.alpha import momentum_signal, moving_average_signal
from src.backtest import StrategyFn
from src.config import (
    ANNUALIZATION_DAYS,
    FACTOR_MODEL_MAX_LOOKBACK,
    MARKOWITZ_DELTA_W,
    MARKOWITZ_TURNOVER_LIMIT,
    MARKOWITZ_W_MAX,
    MIN_HISTORY_DAYS,
    N_STAT_FACTORS,
    TARGET_VOL,
    TRADING_COST_KAPPA,
)
from src.markowitz import solve_markowitz
from src.portfolio_runner import investable_subset
from src.risk_models import fit_statistical_factor_model

# date x ticker alpha value -> {ticker: alpha} at a given (date, row index, active universe)
AlphaFn = Callable[[pd.Timestamp, int, list[str]], dict[str, float]]


def precomputed_alpha_fn(alpha_frame: pd.DataFrame) -> AlphaFn:
    """Wrap a precomputed (date x ticker) alpha panel into an ``AlphaFn``.

    Uses the value as of the previous close (row ``t - 1``), consistent with
    the backtest engine trading at the start of day ``t`` on information
    available through the previous close only (no look-ahead).
    """
    values = alpha_frame.to_numpy(dtype=float)
    col_pos = {c: i for i, c in enumerate(alpha_frame.columns)}

    def fn(date: pd.Timestamp, t: int, active: list[str]) -> dict[str, float]:
        if t == 0:
            return {tk: 0.0 for tk in active}
        row = values[t - 1]
        out = {}
        for tk in active:
            j = col_pos.get(tk)
            v = row[j] if j is not None else np.nan
            out[tk] = 0.0 if (j is None or np.isnan(v)) else float(v)
        return out

    return fn


def momentum_alpha_fn(prices: pd.DataFrame) -> AlphaFn:
    """One-month momentum alpha vector (Sec. 5.2, Sec. 6.3)."""
    return precomputed_alpha_fn(momentum_signal(prices))


def moving_average_alpha_fn(prices: pd.DataFrame) -> AlphaFn:
    """50-day moving-average +/-1 alpha vector (Sec. 5.2, Sec. 6.3)."""
    signal = moving_average_signal(prices)
    frame = pd.DataFrame(signal, index=prices.index, columns=prices.columns)
    return precomputed_alpha_fn(frame)


def realized_future_alpha_fn(returns: pd.DataFrame, horizon: int = ANNUALIZATION_DAYS) -> AlphaFn:
    """Non-implementable regret-bound alpha (Sec. 3.3, Sec. 6.4).

    Uses the average realized daily return over the following ``horizon``
    trading days as the alpha forecast -- deliberately looking ahead to
    obtain an upper bound on what any causal alpha model could achieve. Must
    only be combined with yearly rebalancing so the bound is not inflated by
    unrealistically frequent perfect-foresight trading.
    """
    values = returns.to_numpy(dtype=float)
    col_pos = {c: i for i, c in enumerate(returns.columns)}

    def fn(date: pd.Timestamp, t: int, active: list[str]) -> dict[str, float]:
        window = values[t: t + horizon]
        out = {}
        for tk in active:
            j = col_pos.get(tk)
            if j is None or window.shape[0] == 0:
                out[tk] = 0.0
                continue
            col = window[:, j]
            col = col[~np.isnan(col)]
            out[tk] = float(col.mean()) if len(col) else 0.0
        return out

    return fn


def make_markowitz_strategy_fn(
    alpha_fn: AlphaFn,
    returns: pd.DataFrame,
    listing_index: dict[str, int],
    candidate_tickers: list[str] | None = None,
    n_factors: int = N_STAT_FACTORS,
    sigma_tgt: float = TARGET_VOL,
    turnover_limit: float = MARKOWITZ_TURNOVER_LIMIT,
    delta_w: float = MARKOWITZ_DELTA_W,
    w_max: float = MARKOWITZ_W_MAX,
    kappa: float = TRADING_COST_KAPPA,
    min_history: int = MIN_HISTORY_DAYS,
    annualization_days: int = ANNUALIZATION_DAYS,
    max_lookback: int = FACTOR_MODEL_MAX_LOOKBACK,
) -> StrategyFn:
    """Build a ``StrategyFn`` running the modified Markowitz problem (Sec. 3.3).

    At every rebalance: restrict to tickers investable at ``t``
    (``MIN_HISTORY_DAYS`` of history, Sec. 6.1), fit the statistical factor
    risk model (Sec. 4.3) on the trailing daily-return window strictly
    before ``t``, annualize the resulting factor loadings and idiosyncratic
    variances (``* sqrt(252)`` / ``* 252``, matching the annualized
    ``sigma_tgt``), evaluate ``alpha_fn`` on the same active universe, and
    solve ``solve_markowitz`` for the target weights.
    """
    candidates = candidate_tickers or list(returns.columns)
    returns_np = returns.to_numpy(dtype=float)
    col_pos = {c: i for i, c in enumerate(returns.columns)}

    def strategy_fn(date: pd.Timestamp, t: int, w_pre: dict | None = None) -> dict[str, float]:
        active = investable_subset(candidates, listing_index, t, min_history)
        if len(active) < 2:
            return {}
        idx = [col_pos[tk] for tk in active]
        start = max(0, t - max_lookback)
        window = returns_np[start:t][:, idx]
        if window.shape[0]:
            window = window[~np.isnan(window).any(axis=1)]
        if window.shape[0] < max(20, len(active) + 1):
            return {}

        k = min(n_factors, len(active) - 1)
        factor = fit_statistical_factor_model(window, k=k)
        f_annual = factor.loadings * factor.vol[:, None] * np.sqrt(annualization_days)
        d_annual = (factor.idio_var * factor.vol ** 2) * annualization_days

        alpha_map = alpha_fn(date, t, active)
        alpha = np.array([alpha_map.get(tk, 0.0) for tk in active])
        w_pre_vec = np.array([(w_pre or {}).get(tk, 0.0) for tk in active])

        w = solve_markowitz(
            alpha,
            w_pre_vec,
            factor_loadings=f_annual,
            idio_var=d_annual,
            sigma_tgt=sigma_tgt,
            turnover_limit=turnover_limit,
            delta_w=delta_w,
            w_max=w_max,
            kappa=kappa,
        )
        return dict(zip(active, w))

    return strategy_fn
