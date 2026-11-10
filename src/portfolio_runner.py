"""Wiring between the six simple strategies (Sec. 3.2) and the daily backtest
engine (Sec. 6.1): builds a ``StrategyFn`` that, at each monthly rebalance,
restricts the universe to investable tickers, estimates a trailing covariance
forecast with no look-ahead, and hands it to the requested strategy.

Shared by the exhaustive/targeted simple-strategy search (exp.exp1) and the
final proposed portfolios (exp.exp4), so both experiments reuse one
implementation of "which tickers are investable at date t" and "how the
covariance forecast is estimated" rather than each inventing its own.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.backtest import StrategyFn, is_investable
from src.config import COV_LOOKBACK_L, MIN_HISTORY_DAYS, TARGET_VOL
from src.risk_models import rolling_sample_covariance
from src.strategies import (
    fixed_weights,
    inverse_volatility,
    maximum_diversification,
    minimum_volatility,
    risk_parity,
    volatility_control,
)

METHODS = (
    "fixed",
    "fixed_vc",
    "inverse_vol",
    "min_vol",
    "risk_parity",
    "max_diversification",
)


def listing_day_index(returns: pd.DataFrame) -> dict[str, int]:
    """Row index of the first non-NaN return of each column (ticker)."""
    first_valid = returns.apply(lambda s: s.first_valid_index())
    pos = {d: i for i, d in enumerate(returns.index)}
    return {tkr: pos[d] for tkr, d in first_valid.items() if d is not None}


def investable_subset(
    candidate_tickers: list[str],
    listing_index: dict[str, int],
    t: int,
    min_history: int = MIN_HISTORY_DAYS,
) -> list[str]:
    """Candidate tickers that already have ``min_history`` days of trading at ``t``."""
    return [tk for tk in candidate_tickers if is_investable(listing_index, tk, t, min_history)]


def make_strategy_fn(
    method: str,
    returns: pd.DataFrame,
    listing_index: dict[str, int],
    candidate_tickers: list[str] | None = None,
    fixed_weights_map: dict[str, float] | None = None,
    cov_lookback: int = COV_LOOKBACK_L,
    sigma_tgt: float = TARGET_VOL,
    min_history: int = MIN_HISTORY_DAYS,
) -> StrategyFn:
    """Build a ``StrategyFn`` for one of the six simple strategies (Sec. 3.2).

    Parameters
    ----------
    method : one of ``METHODS``.
    returns : full daily-return panel (dates x tickers); only rows strictly
        before the rebalance date are used to estimate the covariance,
        preventing look-ahead.
    listing_index : output of ``listing_day_index``.
    candidate_tickers : universe to restrict to before filtering by listing
        history; defaults to all columns of ``returns``.
    fixed_weights_map : required for ``method in {"fixed", "fixed_vc"}``,
        predetermined weights by ticker (missing tickers get weight 0).
    """
    assert method in METHODS, f"unknown method {method!r}"
    candidates = candidate_tickers or list(returns.columns)
    if method in ("fixed", "fixed_vc"):
        assert fixed_weights_map is not None, "fixed_weights_map is required for fixed-weight methods"

    def strategy_fn(date: pd.Timestamp, t: int) -> dict[str, float]:
        active = investable_subset(candidates, listing_index, t, min_history)
        if method in ("fixed", "fixed_vc"):
            active = [tk for tk in active if fixed_weights_map.get(tk, 0.0) > 0]
        if len(active) == 0:
            return {}

        if method == "fixed":
            raw = np.array([fixed_weights_map[tk] for tk in active])
            w = fixed_weights(raw / raw.sum())
            return dict(zip(active, w))

        window = returns[active].iloc[max(0, t - cov_lookback):t].fillna(0.0).to_numpy()
        cov = rolling_sample_covariance(window, lookback=min(cov_lookback, max(window.shape[0], 1)))

        if method == "fixed_vc":
            raw = np.array([fixed_weights_map[tk] for tk in active])
            raw = raw / raw.sum()
            w = volatility_control(raw, cov, sigma_tgt=sigma_tgt)
        elif method == "inverse_vol":
            w = inverse_volatility(cov)
        elif method == "min_vol":
            w = minimum_volatility(cov)
        elif method == "risk_parity":
            w = risk_parity(cov)
        elif method == "max_diversification":
            w = maximum_diversification(cov)
        else:  # pragma: no cover - guarded by the assertion above
            raise AssertionError(method)
        return dict(zip(active, w))

    return strategy_fn
