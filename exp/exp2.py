"""Experiment 2: evaluate covariance forecasting methods for ETF portfolio
construction (paper Sec. 4, Sec. 6.3, Table 17/18).

At each month-end forecast date t, three covariance estimators -- rolling
sample covariance, EWMA, and a k=10 statistical factor model -- are fit on
daily returns strictly before t, for each of four evaluation universes
(Equities, Fixed income, Alternatives, Full universe). Each forecast is
compared against the realized covariance of the next 21 trading days via the
normalized Frobenius distance ||Sigma_hat - Sigma_realized||_F / ||Sigma_realized||_F,
averaged across all valid forecast dates (Sec. 4.1/6.3).

The cumulative share of variance explained by the leading principal
components of the full 59-ETF universe is also computed, motivating the
paper's choice of k=10 statistical factors (Sec. 6.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import monthly_rebalance_dates
from src.config import (
    BACKTEST_END,
    BACKTEST_START,
    COV_LOOKBACK_L,
    FACTOR_MODEL_MAX_LOOKBACK,
    MIN_HISTORY_DAYS,
    N_STAT_FACTORS,
    REALIZED_HORIZON_DAYS,
)
from src.market_data import load_universe_panel, restrict_to_sample
from src.risk_models import (
    ewma_covariance,
    explained_variance_by_pc,
    fit_statistical_factor_model,
    rolling_sample_covariance,
)
from src.universe import ASSET_CLASS_OF_CATEGORY, CATEGORIES, FULL_UNIVERSE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METHODS = ("Sample", "EWMA", "K10")


def tickers_of_asset_class(asset_class: str) -> list[str]:
    return [t for cat, tks in CATEGORIES.items() if ASSET_CLASS_OF_CATEGORY[cat] == asset_class for t in tks]


def valid_forecast_positions(n_dates: int, rebalance_positions: list[int]) -> list[int]:
    """Month-end positions with enough trailing history and a full realized window ahead.

    A forecast date t needs at least MIN_HISTORY_DAYS + COV_LOOKBACK_L
    observations strictly before it (Sec. 4.1) and REALIZED_HORIZON_DAYS
    trading days still ahead of it (inclusive of t) to form the realized
    covariance target.
    """
    lo = MIN_HISTORY_DAYS + COV_LOOKBACK_L
    hi = n_dates - REALIZED_HORIZON_DAYS
    return [t for t in rebalance_positions if lo <= t <= hi]


def eligible_tickers_for_period(
    tickers: list[str], listing_index: dict[str, int], first_eval_t: int,
) -> list[str]:
    """Tickers investable for the entire evaluation window (Sec. 6.1's 126-day rule).

    Restricting each universe to this fixed subset, rather than letting
    membership vary date by date, keeps the covariance matrix a fixed size
    across the whole comparison (documented trade-off requested for this
    experiment: simplicity over a time-varying matrix dimension).
    """
    return [tk for tk in tickers if listing_index.get(tk, np.inf) + MIN_HISTORY_DAYS <= first_eval_t]


def frobenius_distance(sigma_hat: np.ndarray, sigma_real: np.ndarray) -> float:
    return float(np.linalg.norm(sigma_hat - sigma_real) / np.linalg.norm(sigma_real))


def evaluate_universe(
    returns: pd.DataFrame,
    listing_index: dict[str, int],
    tickers: list[str],
    forecast_positions: list[int],
) -> tuple[dict[str, float], int, int]:
    """Average normalized Frobenius distance per method for one universe.

    Returns (avg_distance_per_method, n_forecast_dates_used, n_tickers_used).
    """
    first_eval_t = forecast_positions[0]
    eligible = eligible_tickers_for_period(tickers, listing_index, first_eval_t)
    sub = returns[eligible]

    # Narrow universes (e.g. Alternatives, whose 2005-2025-investable subset is
    # a handful of commodity/metal ETFs) can have fewer assets than the paper's
    # k=10 statistical factors; cap k at the matrix dimension so the EM fit
    # stays well-posed instead of crashing on an (n, n) correlation with n < k.
    k_eff = min(N_STAT_FACTORS, len(eligible))

    distances: dict[str, list[float]] = {m: [] for m in METHODS}
    for t in forecast_positions:
        history = sub.iloc[:t]  # strictly before t, no look-ahead

        sample_window = history.iloc[-COV_LOOKBACK_L:].fillna(0.0).to_numpy()
        sigma_sample = rolling_sample_covariance(sample_window, lookback=COV_LOOKBACK_L)

        growing_window = history.iloc[-FACTOR_MODEL_MAX_LOOKBACK:].fillna(0.0).to_numpy()
        sigma_ewma = ewma_covariance(growing_window)
        sigma_factor = fit_statistical_factor_model(growing_window, k=k_eff).covariance()

        realized = sub.iloc[t : t + REALIZED_HORIZON_DAYS].fillna(0.0).to_numpy()
        sigma_real = (realized.T @ realized) / REALIZED_HORIZON_DAYS

        distances["Sample"].append(frobenius_distance(sigma_sample, sigma_real))
        distances["EWMA"].append(frobenius_distance(sigma_ewma, sigma_real))
        distances["K10"].append(frobenius_distance(sigma_factor, sigma_real))

    avg = {m: float(np.mean(d)) for m, d in distances.items()}
    return avg, len(forecast_positions), len(eligible)


def main(full: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    if full:
        start, end = BACKTEST_START, BACKTEST_END
    else:
        start = (pd.Timestamp(BACKTEST_END) - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
        end = BACKTEST_END

    panel = load_universe_panel(FULL_UNIVERSE, start, end, backfill_ibit=True, use_cache=True)
    # IBIT backfill unions the date index back to IBIT_BACKFILL_START (2017-01-01)
    # regardless of ``start``; restrict_to_sample trims the panel back to the
    # requested [start, end] window and recomputes listing_index against that
    # trimmed calendar (a no-op for the full BACKTEST_START=2005 window, since
    # 2005 already precedes 2017, but required for the reduced smoke-test window).
    panel = restrict_to_sample(panel, start, end)
    returns = panel.returns
    listing_index = panel.listing_index
    dates = returns.index

    rebalance_dates = monthly_rebalance_dates(dates)
    pos_of_date = {d: i for i, d in enumerate(dates)}
    rebalance_positions = sorted(pos_of_date[d] for d in rebalance_dates)
    forecast_positions = valid_forecast_positions(len(dates), rebalance_positions)
    assert forecast_positions, "no valid forecast dates in the selected window"

    universes = {
        "Equities": tickers_of_asset_class("Equities"),
        "Fixed income": tickers_of_asset_class("Fixed income"),
        "Alternatives": tickers_of_asset_class("Alternatives"),
        "Full universe": FULL_UNIVERSE,
    }

    rows = {}
    n_dates_per_universe = {}
    for name, tickers in universes.items():
        avg, n_dates, n_tickers = evaluate_universe(returns, listing_index, tickers, forecast_positions)
        rows[name] = avg
        n_dates_per_universe[name] = (n_dates, n_tickers)

    results = pd.DataFrame(rows).T[list(METHODS)]
    results.to_csv(RESULTS_DIR / "exp2_covariance_distance.csv")

    full_returns = returns[FULL_UNIVERSE].dropna(how="any")
    cum_var = explained_variance_by_pc(full_returns.to_numpy())
    ev_df = pd.DataFrame(
        {"n_components": np.arange(1, len(cum_var) + 1), "cumulative_variance_explained": cum_var}
    )
    ev_df.to_csv(RESULTS_DIR / "exp2_explained_variance.csv", index=False)

    n_plot = min(30, len(cum_var))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, n_plot + 1), cum_var[:n_plot], marker="o", markersize=3)
    ax.set_title("Cumulative variance explained by principal components\n(full 59-ETF universe)")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative share of variance explained")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_explained_variance.png", dpi=150)
    plt.close(fig)

    print(f"Window: {start} to {end} ({'full' if full else 'reduced'})")
    print("Forecast dates evaluated per universe (n_dates, n_tickers_used):")
    for name, (n_dates, n_tickers) in n_dates_per_universe.items():
        print(f"  {name}: {n_dates} dates, {n_tickers} tickers")
    print("\nNormalized Frobenius distance (lower is better):")
    print(results)
    print(f"\nExplained variance: PC1={cum_var[0]:.4f}, PC10={cum_var[9] if len(cum_var) > 9 else cum_var[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full paper backtest window (BACKTEST_START to BACKTEST_END); "
        "default uses a faster reduced (last 6 years) window for smoke testing.",
    )
    args = parser.parse_args()
    main(full=args.full)
