"""Experiment 4: proposed final portfolios and benchmark robustness across
full and truncated backtest windows (paper Sec. 6.4, Table 25, Fig. 26).

Builds and backtests six portfolios over one shared ``UniversePanel`` loaded
on ``FULL_UNIVERSE`` (a superset of every ticker any of the six needs):

  1. 60/40 benchmark (fixed SPY/AGG, monthly, no vol control)
  2. Buy-and-hold SPY, 8% volatility control ("SPY vc8%")
  3. SPY/AGG/GLD equal weights (fixed, monthly, no vol control)
  4. SPY/TLT/GLD/IBIT maximum diversification
  5. Momentum Universe Markowitz (momentum alpha, post-ablation universe)
  6. Regret bound: Markowitz with realized-future alpha over the full
     59-ETF universe, rebalanced yearly -- a non-implementable upper bound,
     never a real strategy (Sec. 3.3/6.4).

Also repeats the five *implementable* portfolios (excluding the regret
bound) over the two truncated windows of Sec. 6.4: excluding the first five
years (2008 Financial Crisis) and excluding the last five years (COVID-19,
2022 drawdown), built from the same loaded panel via ``restrict_to_sample``.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import BacktestResult, run_backtest
from src.config import (
    ANNUALIZATION_DAYS,
    BACKTEST_END,
    BACKTEST_START,
    BENCHMARK_6040,
    MOMENTUM_UNIVERSE_REMOVED_CATEGORIES,
    TARGET_VOL,
    TRUNCATED_WINDOW_EXCL_FIRST5,
    TRUNCATED_WINDOW_EXCL_LAST5,
)
from src.market_data import UniversePanel, load_universe_panel, restrict_to_sample
from src.markowitz_runner import (
    make_markowitz_strategy_fn,
    momentum_alpha_fn,
    realized_future_alpha_fn,
)
from src.metrics import performance_summary
from src.portfolio_runner import listing_day_index, make_strategy_fn
from src.universe import FULL_UNIVERSE, universe_excluding_categories

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SMOKE_WINDOW_YEARS = 6  # smoke-test window length when --full is not given

REGRET_BOUND_NAME = "regret_bound"
REGRET_BOUND_LABEL = "regret_bound (NOT IMPLEMENTABLE -- uses realized future returns)"


def summarize(result: BacktestResult) -> dict[str, float]:
    """Performance summary of a backtest, keyed as in the paper's tables (eq. 28)."""
    out = performance_summary(result.nav, result.portfolio_returns, result.excess_returns, result.turnover)
    return {
        "r_e": out["r"], "sigma": out["sigma"], "MDD": out["MDD"],
        "TO": out["TO"], "SR": out["SR"], "LQ": out["LQ"],
    }


def _drop_unlisted_in_window(panel: UniversePanel, start: str, end: str) -> UniversePanel:
    """Drop tickers with zero valid observations within ``[start, end]``.

    ``src.portfolio_runner.listing_day_index`` (used inside
    ``src.market_data.restrict_to_sample``) assumes every column has at
    least one non-NaN value: ``returns.apply(lambda s: s.first_valid_index())``
    coerces an all-NaN column's ``None`` result to ``NaT`` once other
    columns return Timestamps (forcing a datetime64 dtype on the whole
    row), and the existing ``d is not None`` guard does not catch ``NaT``,
    so looking it up in the date->position map raises ``KeyError: NaT``.
    This hits tickers such as ETHA (listed 2024) inside the "excl_last_5y"
    robustness window (ending 2020-01-01), where they have no history at
    all. Those tickers are never candidates for any of the six portfolios
    in that window anyway (they would fail ``MIN_HISTORY_DAYS`` regardless),
    so dropping them here before calling ``restrict_to_sample`` changes
    nothing about the backtests and only avoids the crash.
    """
    mask = (panel.prices.index >= pd.Timestamp(start)) & (panel.prices.index <= pd.Timestamp(end))
    window_returns = panel.returns.loc[mask]
    valid_cols = [c for c in panel.returns.columns if window_returns[c].notna().any()]
    if len(valid_cols) == len(panel.returns.columns):
        return panel
    dropped = sorted(set(panel.returns.columns) - set(valid_cols))
    print(f"  (window {start}..{end}: no observations at all for {dropped}; excluded before restrict_to_sample)")
    return UniversePanel(
        prices=panel.prices[valid_cols],
        returns=panel.returns[valid_cols],
        listing_index=listing_day_index(panel.returns[valid_cols]),
        ffr_daily=panel.ffr_daily,
        meta=panel.meta,
    )


def yearly_rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar year within ``dates`` (Sec. 6.4 regret bound).

    ``src.backtest`` only ships ``monthly_rebalance_dates``; the paper
    restricts the regret-bound portfolio to yearly trading so its perfect-
    foresight alpha is not inflated by unrealistically frequent rebalancing
    (Sec. 3.3/6.4). This mirrors ``monthly_rebalance_dates``'s own convention
    (last available trading day of the period) at yearly granularity.
    """
    s = pd.Series(dates, index=dates)
    last_of_year = s.groupby(dates.year).max()
    return pd.DatetimeIndex(sorted(last_of_year.values))


# ---------------------------------------------------------------------------
# Portfolio builders: each maps (panel, momentum_universe) -> BacktestResult
# ---------------------------------------------------------------------------

def build_6040(panel: UniversePanel, momentum_universe: list[str]) -> BacktestResult:
    strategy = make_strategy_fn(
        "fixed", panel.returns, panel.listing_index,
        candidate_tickers=list(BENCHMARK_6040), fixed_weights_map=BENCHMARK_6040,
    )
    return run_backtest(panel.returns, strategy, panel.ffr_daily)


def build_spy_vc8(panel: UniversePanel, momentum_universe: list[str]) -> BacktestResult:
    weights_map = {"SPY": 1.0}
    strategy = make_strategy_fn(
        "fixed_vc", panel.returns, panel.listing_index,
        candidate_tickers=list(weights_map), fixed_weights_map=weights_map, sigma_tgt=TARGET_VOL,
    )
    return run_backtest(panel.returns, strategy, panel.ffr_daily)


def build_equal_spy_agg_gld(panel: UniversePanel, momentum_universe: list[str]) -> BacktestResult:
    weights_map = {"SPY": 1.0 / 3, "AGG": 1.0 / 3, "GLD": 1.0 / 3}
    strategy = make_strategy_fn(
        "fixed", panel.returns, panel.listing_index,
        candidate_tickers=list(weights_map), fixed_weights_map=weights_map,
    )
    return run_backtest(panel.returns, strategy, panel.ffr_daily)


def build_maxdiv_spy_tlt_gld_ibit(panel: UniversePanel, momentum_universe: list[str]) -> BacktestResult:
    tickers = [tk for tk in ("SPY", "TLT", "GLD", "IBIT") if tk in panel.returns.columns]
    strategy = make_strategy_fn(
        "max_diversification", panel.returns, panel.listing_index, candidate_tickers=tickers,
    )
    return run_backtest(panel.returns, strategy, panel.ffr_daily)


def build_momentum_universe_markowitz(panel: UniversePanel, momentum_universe: list[str]) -> BacktestResult:
    candidates = [tk for tk in momentum_universe if tk in panel.returns.columns]
    alpha_fn = momentum_alpha_fn(panel.prices[candidates])
    strategy = make_markowitz_strategy_fn(
        alpha_fn, panel.returns, panel.listing_index, candidate_tickers=candidates,
    )
    return run_backtest(panel.returns, strategy, panel.ffr_daily)


def build_regret_bound(panel: UniversePanel, momentum_universe: list[str]) -> BacktestResult:
    candidates = [tk for tk in FULL_UNIVERSE if tk in panel.returns.columns]
    alpha_fn = realized_future_alpha_fn(panel.returns, horizon=ANNUALIZATION_DAYS)
    strategy = make_markowitz_strategy_fn(
        alpha_fn, panel.returns, panel.listing_index, candidate_tickers=candidates,
    )
    rebalance_dates = yearly_rebalance_dates(panel.returns.index)
    return run_backtest(panel.returns, strategy, panel.ffr_daily, rebalance_dates=rebalance_dates)


# Order matches the paper's Table 25 listing (1-6); dict preserves insertion order.
IMPLEMENTABLE_BUILDERS = {
    "60_40_benchmark": build_6040,
    "spy_vc8": build_spy_vc8,
    "spy_agg_gld_equal": build_equal_spy_agg_gld,
    "spy_tlt_gld_ibit_maxdiv": build_maxdiv_spy_tlt_gld_ibit,
    "momentum_universe_markowitz": build_momentum_universe_markowitz,
}
ALL_BUILDERS = {**IMPLEMENTABLE_BUILDERS, REGRET_BOUND_NAME: build_regret_bound}


def label_of(name: str) -> str:
    return REGRET_BOUND_LABEL if name == REGRET_BOUND_NAME else name


def plot_cumulative_growth(results: dict[str, BacktestResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, result in results.items():
        ax.plot(result.nav.index, result.nav.values, label=label_of(name))
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative NAV (log scale, start = 1.0)")
    ax.set_title("Experiment 4: cumulative growth of the final portfolios (Sec. 6.4, Fig. 26)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_builders(
    builders: dict, panel: UniversePanel, momentum_universe: list[str],
) -> dict[str, BacktestResult]:
    results = {}
    for name, builder in builders.items():
        t0 = time.time()
        results[name] = builder(panel, momentum_universe)
        print(f"  {name} done in {time.time() - t0:.1f}s")
    return results


def main(full: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    start = BACKTEST_START if full else str(
        (pd.Timestamp(BACKTEST_END) - pd.DateOffset(years=SMOKE_WINDOW_YEARS)).date()
    )
    end = BACKTEST_END
    print(f"Loading universe panel: {len(FULL_UNIVERSE)} tickers, {start} to {end}...")
    t0 = time.time()
    panel = load_universe_panel(FULL_UNIVERSE, start, end)
    print(
        f"Loaded {panel.returns.shape[1]} tickers, {panel.returns.shape[0]} rows "
        f"in {time.time() - t0:.1f}s, {panel.returns.index.min().date()} to {panel.returns.index.max().date()}"
    )

    momentum_universe = [
        tk for tk in universe_excluding_categories(MOMENTUM_UNIVERSE_REMOVED_CATEGORIES)
        if tk in panel.returns.columns
    ]

    print("Building and backtesting the six final portfolios on the full window...")
    results = run_builders(ALL_BUILDERS, panel, momentum_universe)

    rows = []
    for name, result in results.items():
        row = {"portfolio": label_of(name)}
        row.update(summarize(result))
        rows.append(row)
    full_df = pd.DataFrame(rows, columns=["portfolio", "r_e", "sigma", "MDD", "TO", "SR", "LQ"])
    full_df.to_csv(RESULTS_DIR / "exp4_final_portfolios_full.csv", index=False)
    print(full_df.to_string(index=False))

    warnings.warn(
        "'regret_bound' uses realized future returns (perfect foresight) and is NOT an "
        "implementable strategy; it is reported only as a theoretical upper bound on "
        "what any causal alpha model could achieve (Sec. 3.3/6.4).",
        UserWarning,
        stacklevel=1,
    )
    print(f"WARNING: {REGRET_BOUND_LABEL} -- never present this row as a real strategy.")

    plot_cumulative_growth(results, RESULTS_DIR / "exp4_cumulative_growth.png")

    if full:
        robustness_rows = []
        windows = {
            "excl_first_5y": TRUNCATED_WINDOW_EXCL_FIRST5,
            "excl_last_5y": TRUNCATED_WINDOW_EXCL_LAST5,
        }
        for window_name, (w_start, w_end) in windows.items():
            print(f"Robustness window '{window_name}': {w_start} to {w_end}")
            safe_panel = _drop_unlisted_in_window(panel, w_start, w_end)
            sub_panel = restrict_to_sample(safe_panel, w_start, w_end)
            sub_momentum_universe = [tk for tk in momentum_universe if tk in sub_panel.returns.columns]
            sub_results = run_builders(IMPLEMENTABLE_BUILDERS, sub_panel, sub_momentum_universe)
            for name, result in sub_results.items():
                row = {"window": window_name, "portfolio": name}
                row.update(summarize(result))
                robustness_rows.append(row)
        robustness_df = pd.DataFrame(
            robustness_rows, columns=["window", "portfolio", "r_e", "sigma", "MDD", "TO", "SR", "LQ"],
        )
        robustness_df.to_csv(RESULTS_DIR / "exp4_robustness_windows.csv", index=False)
        print(robustness_df.to_string(index=False))
    else:
        print("Smoke mode (--full not set): skipping truncated-window robustness checks.")

    print("Experiment 4 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="run the paper-scale configuration: full 2005-2025 window for all six "
             "portfolios plus both truncated-window robustness checks for the five "
             "implementable ones; otherwise runs a reduced recent window as a smoke test",
    )
    args = parser.parse_args()
    main(full=args.full)
