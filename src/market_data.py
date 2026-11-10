"""Shared data-preparation path from raw bars to the panels every experiment
consumes (paper Sec. 6.1, 6.2).

All four experiments build their investable universe and return panel through
``load_universe_panel`` so there is exactly one implementation of "which
ETFs are in scope", "how IBIT's pre-listing history is reconstructed from
bitcoin" and "which days are investable" (Sec. 6.1's 126-day minimum history
rule, applied uniformly through ``portfolio_runner.listing_day_index``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.data_loader import (
    daily_cash_return,
    load_adjusted_close_panel,
    load_bitcoin_backfill,
    load_fed_funds_rate,
)
from src.config import IBIT_BACKFILL_START
from src.portfolio_runner import listing_day_index


@dataclass
class UniversePanel:
    prices: pd.DataFrame        # date x ticker synthetic/adjusted close level (returns-consistent)
    returns: pd.DataFrame       # date x ticker daily simple returns, NaN before first valid price
    listing_index: dict[str, int]
    ffr_daily: pd.Series        # daily cash return r_f,t aligned to prices.index
    meta: dict


def backfill_ibit_returns(ibit_prices: pd.Series, btc_prices: pd.Series) -> pd.Series:
    """Daily return series for IBIT, extended backward with bitcoin spot returns.

    Sec. 6.2 extends IBIT backward with bitcoin spot returns from 2017; the
    exact source/convention is not specified on the retrieved pages (recorded
    in results/RESULTS.md), so this project uses bitcoin's own daily return
    series as a direct proxy for IBIT's pre-listing daily return, joined at
    IBIT's first real trading date so the two segments chain without a
    fabricated jump at the splice point.
    """
    ibit_returns = ibit_prices.pct_change()
    first_real = ibit_returns.first_valid_index()
    assert first_real is not None, "IBIT has no valid price observation"
    btc_returns = btc_prices.pct_change()
    pre = btc_returns[btc_returns.index < first_real]
    post = ibit_returns[ibit_returns.index >= first_real]
    full = pd.concat([pre, post]).sort_index()
    return full[~full.index.duplicated(keep="last")]


def _returns_to_price_index(returns: pd.Series, base: float = 100.0) -> pd.Series:
    """Synthetic price level consistent with a return series (first day flat)."""
    r = returns.fillna(0.0)
    return base * (1.0 + r).cumprod()


def load_universe_panel(
    tickers: list[str],
    start_date: str,
    end_date: str,
    identity_date: str | None = None,
    backfill_ibit: bool = True,
    use_cache: bool = True,
) -> UniversePanel:
    """Load prices, returns, listing index and the cash return for ``tickers``.

    Parameters mirror Sec. 6.1: ``identity_date`` defaults to ``end_date`` (no
    point-in-time universe snapshot date is given in the paper beyond the
    fixed Table 1 list), and IBIT is backfilled with bitcoin spot returns from
    ``IBIT_BACKFILL_START`` (Sec. 6.2) whenever it is part of ``tickers`` and
    ``backfill_ibit`` is True.
    """
    identity_date = identity_date or end_date
    prices, meta = load_adjusted_close_panel(
        tickers, start_date, end_date, identity_date=identity_date, use_cache=use_cache,
    )
    returns = prices.pct_change()

    if backfill_ibit and "IBIT" in prices.columns:
        btc_prices, btc_meta = load_adjusted_close_panel(
            ["X:BTCUSD"], IBIT_BACKFILL_START, end_date, yahoo_backstop=False, use_cache=use_cache,
        )
        ibit_full_returns = backfill_ibit_returns(prices["IBIT"], btc_prices["X:BTCUSD"])
        ibit_full_returns = ibit_full_returns.reindex(
            prices.index.union(ibit_full_returns.index)
        ).loc[lambda s: s.index >= pd.Timestamp(IBIT_BACKFILL_START)]
        combined_index = prices.index.union(ibit_full_returns.index).sort_values()
        returns = returns.reindex(combined_index)
        prices = prices.reindex(combined_index)
        returns["IBIT"] = ibit_full_returns.reindex(combined_index)
        meta["ibit_backfill"] = {
            "start": IBIT_BACKFILL_START,
            "source_ticker": "X:BTCUSD",
            "btc_ticker_details": btc_meta.get("ticker_details"),
        }

    listing_index = listing_day_index(returns)
    ffr_annual = load_fed_funds_rate(prices.index.min().strftime("%Y-%m-%d"), end_date, use_cache=use_cache)
    ffr_daily = daily_cash_return(ffr_annual, prices.index)

    return UniversePanel(
        prices=prices, returns=returns, listing_index=listing_index, ffr_daily=ffr_daily, meta=meta,
    )


def restrict_to_sample(panel: UniversePanel, start_date: str, end_date: str) -> UniversePanel:
    """Slice an already-loaded panel to a truncated window (Sec. 6.4 robustness windows).

    Listing indices are recomputed against the truncated calendar so
    ``is_investable`` still measures history relative to the window actually
    backtested, not the full load.
    """
    mask = (panel.prices.index >= pd.Timestamp(start_date)) & (panel.prices.index <= pd.Timestamp(end_date))
    prices = panel.prices.loc[mask]
    returns = panel.returns.loc[mask]
    listing_index = listing_day_index(returns)
    ffr_daily = panel.ffr_daily.loc[mask]
    return UniversePanel(
        prices=prices, returns=returns, listing_index=listing_index, ffr_daily=ffr_daily, meta=panel.meta,
    )
