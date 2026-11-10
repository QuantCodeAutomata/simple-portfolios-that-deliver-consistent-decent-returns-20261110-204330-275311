"""Experiment 3: alpha models and Markowitz portfolios on the full ETF
universe, with category ablations (paper Sec. 5.2, 5.3, 6.3, Table 19-24).

Part A (Sec. 5.3, Table 19/20, Fig. 25) sweeps the ridge penalty lambda and
reports the rolling out-of-sample R^2 of the per-asset ridge alpha model,
averaged across the universe.

Part B (Sec. 6.3, Table 21-24) runs the modified long-only Markowitz
portfolio (``src.markowitz_runner`` + ``src.backtest``) with the momentum and
moving-average alpha signals on the full 59-ETF universe, then on the
universe with each of the nine Table 2 categories removed one at a time, then
on the paper's post-ablation "Momentum Universe" / "MovingAverage Universe".
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.alpha import rolling_ridge_oos_r2
from src.backtest import BacktestResult, run_backtest
from src.config import (
    BACKTEST_END,
    BACKTEST_START,
    LTR_WINDOW,
    MOMENTUM_UNIVERSE_REMOVED_CATEGORIES,
    MOVAVG_UNIVERSE_REMOVED_CATEGORIES,
    RIDGE_FORECAST_HORIZON,
    RIDGE_LAMBDA_GRID,
    RIDGE_LOOKBACK_H,
)
from src.market_data import load_universe_panel
from src.markowitz_runner import (
    make_markowitz_strategy_fn,
    momentum_alpha_fn,
    moving_average_alpha_fn,
)
from src.metrics import performance_summary
from src.portfolio_runner import listing_day_index
from src.universe import CATEGORIES, FULL_UNIVERSE, universe_excluding_categories

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ALPHA_FNS = {"momentum": momentum_alpha_fn, "moving_average": moving_average_alpha_fn}
RIDGE_MIN_ROWS = RIDGE_LOOKBACK_H + RIDGE_FORECAST_HORIZON + LTR_WINDOW[1]  # H_ridge + h + 756
SMOKE_RIDGE_N_TICKERS = 15
SMOKE_CATEGORIES = ("Market", "Sector", "Country")
SMOKE_WINDOW_YEARS = 5


def summarize(result: BacktestResult) -> dict[str, float]:
    """Performance summary of a backtest, keyed as in the paper's tables (eq. 28)."""
    out = performance_summary(result.nav, result.portfolio_returns, result.excess_returns, result.turnover)
    return {
        "r_e": out["r"], "sigma": out["sigma"], "MDD": out["MDD"],
        "TO": out["TO"], "SR": out["SR"], "LQ": out["LQ"],
    }


# ---------------------------------------------------------------------------
# Part A: ridge regression out-of-sample R^2 sweep (Sec. 5.3)
# ---------------------------------------------------------------------------

def ridge_r2_sweep(
    tickers: list[str],
    prices: pd.DataFrame,
    lambdas: list[float] = RIDGE_LAMBDA_GRID,
    min_rows: int = RIDGE_MIN_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rolling-ridge OOS R^2 for each (ticker, lambda), skipping short histories.

    Returns (per_asset_long, aggregated_by_lambda, skipped_tickers).
    ``per_asset_long`` has columns ticker, lambda, r2. ``aggregated_by_lambda``
    has columns lambda, mean_r2, std_r2, n_assets.
    """
    rows = []
    skipped = []
    for ticker in tickers:
        series = prices[ticker].dropna()
        if len(series) < min_rows:
            skipped.append(ticker)
            continue
        ticker_failed = False
        ticker_rows = []
        for lam in lambdas:
            try:
                r2 = rolling_ridge_oos_r2(series, lam)
            except AssertionError:
                ticker_failed = True
                break
            ticker_rows.append({"ticker": ticker, "lambda": lam, "r2": r2})
        if ticker_failed:
            skipped.append(ticker)
            continue
        rows.extend(ticker_rows)

    per_asset = pd.DataFrame(rows, columns=["ticker", "lambda", "r2"])
    agg = (
        per_asset.groupby("lambda")["r2"]
        .agg(mean_r2="mean", std_r2="std", n_assets="count")
        .reset_index()
        .sort_values("lambda")
        .reset_index(drop=True)
    )
    agg["std_r2"] = agg["std_r2"].fillna(0.0)
    return per_asset, agg, skipped


def plot_ridge_r2(agg: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.log10(agg["lambda"])
    yerr = agg["std_r2"] / np.sqrt(agg["n_assets"].clip(lower=1))
    ax.errorbar(x, agg["mean_r2"], yerr=yerr, marker="o", capsize=3)
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("Mean out-of-sample R^2 across assets")
    ax.set_title("Rolling ridge regression: OOS R^2 vs. penalty (Sec. 5.3, Fig. 25)")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_part_a(panel, full: bool, seed: int = 42) -> None:
    if full:
        tickers = list(FULL_UNIVERSE)
    else:
        rng = np.random.default_rng(seed)
        tickers = list(rng.choice(FULL_UNIVERSE, size=min(SMOKE_RIDGE_N_TICKERS, len(FULL_UNIVERSE)), replace=False))

    print(f"Part A: ridge R^2 sweep over {len(tickers)} tickers, {len(RIDGE_LAMBDA_GRID)} lambdas...")
    t0 = time.time()
    per_asset, agg, skipped = ridge_r2_sweep(tickers, panel.prices)
    print(f"Part A done in {time.time() - t0:.1f}s; skipped (insufficient history): {skipped}")

    agg.to_csv(RESULTS_DIR / "exp3_ridge_r2.csv", index=False)
    plot_ridge_r2(agg, RESULTS_DIR / "exp3_ridge_r2.png")
    return per_asset, agg, skipped


# ---------------------------------------------------------------------------
# Part B: Markowitz full-universe backtest + category ablation (Sec. 6.3)
# ---------------------------------------------------------------------------

def restrict_panel(panel, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Column-restrict ``panel``'s returns/prices to ``tickers`` already present in it."""
    cols = [t for t in tickers if t in panel.returns.columns]
    returns = panel.returns[cols]
    prices = panel.prices[cols]
    listing_index = listing_day_index(returns)
    return returns, prices, listing_index


def run_markowitz_row(
    signal: str,
    alpha_fn_builder,
    universe: list[str],
    label: str,
    panel,
) -> dict:
    returns, prices, listing_index = restrict_panel(panel, universe)
    alpha_fn = alpha_fn_builder(prices)
    strategy_fn = make_markowitz_strategy_fn(
        alpha_fn, returns, listing_index, candidate_tickers=list(returns.columns),
    )
    result = run_backtest(returns, strategy_fn, panel.ffr_daily)
    row = {"signal": signal, "universe": label, "n_tickers": returns.shape[1]}
    row.update(summarize(result))
    return row


def run_part_b(panel, full: bool) -> pd.DataFrame:
    categories = list(CATEGORIES) if full else list(SMOKE_CATEGORIES)
    post_ablation_categories = {
        "momentum": MOMENTUM_UNIVERSE_REMOVED_CATEGORIES,
        "moving_average": MOVAVG_UNIVERSE_REMOVED_CATEGORIES,
    }
    post_ablation_label = {
        "momentum": "Momentum Universe",
        "moving_average": "MovingAverage Universe",
    }

    rows = []
    for signal, alpha_fn_builder in ALPHA_FNS.items():
        t0 = time.time()
        rows.append(run_markowitz_row(signal, alpha_fn_builder, FULL_UNIVERSE, "Full universe", panel))
        print(f"  [{signal}] Full universe done ({time.time() - t0:.1f}s)")

        for cat in categories:
            t0 = time.time()
            universe = universe_excluding_categories((cat,))
            rows.append(run_markowitz_row(signal, alpha_fn_builder, universe, f"minus_{cat}", panel))
            print(f"  [{signal}] minus_{cat} done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        removed = post_ablation_categories[signal]
        universe = universe_excluding_categories(removed)
        label = post_ablation_label[signal]
        rows.append(run_markowitz_row(signal, alpha_fn_builder, universe, label, panel))
        print(f"  [{signal}] {label} (removed {removed}) done ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows, columns=["signal", "universe", "n_tickers", "r_e", "sigma", "MDD", "TO", "SR", "LQ"])
    df.to_csv(RESULTS_DIR / "exp3_markowitz_ablation.csv", index=False)
    return df


def main(full: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    start = BACKTEST_START if full else str((pd.Timestamp(BACKTEST_END) - pd.DateOffset(years=SMOKE_WINDOW_YEARS)).date())
    print(f"Loading universe panel: {len(FULL_UNIVERSE)} tickers, {start} to {BACKTEST_END}...")
    panel = load_universe_panel(FULL_UNIVERSE, start, BACKTEST_END)
    print(f"Loaded {panel.returns.shape[1]} tickers, {panel.returns.shape[0]} rows, "
          f"{panel.returns.index.min().date()} to {panel.returns.index.max().date()}")

    run_part_a(panel, full=full)

    print("Part B: Markowitz full-universe backtest + category ablation...")
    t0 = time.time()
    ablation_df = run_part_b(panel, full=full)
    print(f"Part B done in {time.time() - t0:.1f}s")
    print(ablation_df.to_string(index=False))

    print("Experiment 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="run the paper-scale configuration: full 2005-2025 window, all FULL_UNIVERSE "
             "tickers with enough history in Part A, all nine category ablations in Part B",
    )
    args = parser.parse_args()
    main(full=args.full)
