"""Experiment 1: simple long-only portfolio-construction strategies on small
ETF subsets (paper Sec. 6.2).

Runs, in order: (1) the exhaustive equal-weight search over the filtered
23-ETF universe, ranked by Sharpe ratio and by Sharpe ratio then LQ; (2) the
SPY/AGG/GLD fixed-weight split grid, without and with 8% volatility control,
compared to the 60/40 benchmark; (3) inverse-volatility sizing applied to the
best equal-weight portfolios and an independent exhaustive inverse-volatility
search; (4) targeted minimum-volatility / risk-parity / maximum-diversification
diagnostics (duplicate-asset, complementary-asset, extreme-asset/IBIT tests);
(5) the maximum-diversification broad-universe test; (6) the strongest
hindsight-selected portfolio found across all of the above.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.config import (
    BACKTEST_END,
    BACKTEST_START,
    BENCHMARK_6040,
    EXHAUSTIVE_SEARCH_MAX_COMBOS_RUN,
    FIXED_SPLIT_GRID_STEP,
    SIMPLE_SEARCH_PORTFOLIO_SIZES,
)
from src.exhaustive_search import rank_by_sr, rank_by_sr_then_lq, search_combinations
from src.market_data import load_universe_panel
from src.metrics import performance_summary
from src.portfolio_runner import make_strategy_fn
from src.universe import CATEGORIES, FULL_UNIVERSE, SEARCH_UNIVERSE_23

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LISTED_AT_START_TOLERANCE = 5  # rows of initial NaN still counted as "listed at the start" (Sec. 6.2)


def _summary(returns, listing_index, ffr_daily, candidates, method, fixed_weights_map=None):
    strategy = make_strategy_fn(
        method, returns, listing_index, candidate_tickers=candidates, fixed_weights_map=fixed_weights_map,
    )
    result = run_backtest(returns, strategy, ffr_daily)
    out = performance_summary(result.nav, result.portfolio_returns, result.excess_returns, result.turnover)
    out["avg_weights"] = {
        tk: float(result.weights[tk][result.weights[tk] > 1e-6].mean()) if (result.weights[tk] > 1e-6).any() else 0.0
        for tk in candidates
    }
    return out


def main(full: bool) -> None:
    max_combos = None if full else EXHAUSTIVE_SEARCH_MAX_COMBOS_RUN
    RESULTS_DIR.mkdir(exist_ok=True)

    panel = load_universe_panel(FULL_UNIVERSE, BACKTEST_START, BACKTEST_END)
    returns, listing_index, ffr_daily = panel.returns, panel.listing_index, panel.ffr_daily
    print(f"Loaded {returns.shape[1]} tickers, {returns.shape[0]} rows, "
          f"{returns.index.min().date()} to {returns.index.max().date()}")

    log: dict = {
        "n_tickers_loaded": returns.shape[1],
        "n_rows": int(returns.shape[0]),
        "date_range": [str(returns.index.min().date()), str(returns.index.max().date())],
        "non_massive_source": panel.meta.get("non_massive_source", {}),
        "warnings": panel.meta.get("warnings", {}),
        "failed": panel.meta.get("failed", {}),
    }

    # --- Sec 6.2: 23-ETF filtered search universe (exclude Country/Futures,
    # then exclude ETFs not already listed at the start of the backtest). ---
    filtered_23 = [tk for tk in SEARCH_UNIVERSE_23 if listing_index.get(tk, 10 ** 9) <= LISTED_AT_START_TOLERANCE]
    log["search_universe_23_candidates_before_listing_filter"] = SEARCH_UNIVERSE_23
    log["search_universe_23_filtered"] = filtered_23
    log["search_universe_23_filtered_count"] = len(filtered_23)
    print(f"Filtered 23-ETF search universe: {len(filtered_23)} ETFs listed at the start of the backtest")

    # 60/40 benchmark -----------------------------------------------------
    benchmark = _summary(returns, listing_index, ffr_daily, list(BENCHMARK_6040), "fixed", BENCHMARK_6040)

    # ---------------------------------------------------------------------
    # 1) Exhaustive equal-weight search (Sec. 6.2, criterion (i) and (ii))
    # ---------------------------------------------------------------------
    print("Running exhaustive equal-weight search...")
    eq_results = search_combinations(
        returns, filtered_23, SIMPLE_SEARCH_PORTFOLIO_SIZES, method="equal", max_combos=max_combos,
    )
    eq_results.to_csv(RESULTS_DIR / "exp1_equal_weight_search_raw.csv", index=False)
    eq_by_sr = rank_by_sr(eq_results)
    eq_by_sr_lq = rank_by_sr_then_lq(eq_results)
    eq_by_sr.head(20).to_csv(RESULTS_DIR / "exp1_equal_weight_top20_by_SR.csv", index=False)
    eq_by_sr_lq.head(20).to_csv(RESULTS_DIR / "exp1_equal_weight_top20_by_SR_then_LQ.csv", index=False)

    # Re-evaluate the top-5 of each ranking with the exact daily engine.
    top_equal_rows = []
    seen = set()
    for df in (eq_by_sr.head(5), eq_by_sr_lq.head(5)):
        for _, row in df.iterrows():
            tickers = tuple(row["tickers"])
            if tickers in seen:
                continue
            seen.add(tickers)
            weights_map = {tk: 1.0 / len(tickers) for tk in tickers}
            exact = _summary(returns, listing_index, ffr_daily, list(tickers), "fixed", weights_map)
            top_equal_rows.append({"tickers": tickers, **{k: v for k, v in exact.items() if k != "avg_weights"}})
    top_equal_df = pd.DataFrame(top_equal_rows).sort_values("SR", ascending=False)
    top_equal_df.to_csv(RESULTS_DIR / "exp1_equal_weight_top_exact.csv", index=False)

    # Recurring ETF / category frequency among the best screened candidates.
    category_counts = pd.Series(
        [tk for tks in eq_by_sr.head(200)["tickers"] for tk in tks]
    ).value_counts()
    category_counts.to_csv(RESULTS_DIR / "exp1_equal_weight_recurring_tickers.csv")

    # ---------------------------------------------------------------------
    # 2) SPY/AGG/GLD fixed-weight split grid (Sec. 6.2)
    # ---------------------------------------------------------------------
    print("Running SPY/AGG/GLD split grid...")
    step = FIXED_SPLIT_GRID_STEP
    n_steps = round(1.0 / step)
    split_rows = []
    for i, j in itertools.product(range(n_steps + 1), repeat=2):
        if i + j > n_steps:
            continue
        w_spy, w_agg, w_gld = i * step, j * step, 1.0 - i * step - j * step
        weights_map = {"SPY": w_spy, "AGG": w_agg, "GLD": w_gld}
        weights_map = {k: v for k, v in weights_map.items() if v > 1e-9}
        plain = _summary(returns, listing_index, ffr_daily, list(weights_map), "fixed", weights_map)
        vc = _summary(returns, listing_index, ffr_daily, list(weights_map), "fixed_vc", weights_map)
        split_rows.append({"SPY": w_spy, "AGG": w_agg, "GLD": w_gld, "vc": False,
                            **{k: v for k, v in plain.items() if k != "avg_weights"}})
        split_rows.append({"SPY": w_spy, "AGG": w_agg, "GLD": w_gld, "vc": True,
                            **{k: v for k, v in vc.items() if k != "avg_weights"}})
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(RESULTS_DIR / "exp1_spy_agg_gld_split_grid.csv", index=False)
    best_split = split_df.loc[split_df["SR"].idxmax()]

    # ---------------------------------------------------------------------
    # 3) Inverse volatility: on the best equal-weight portfolios, and a
    #    broader exhaustive inverse-volatility search (Sec. 6.2).
    # ---------------------------------------------------------------------
    print("Running inverse-volatility tests...")
    invvol_on_best_rows = []
    for tickers in top_equal_df["tickers"].head(10):
        exact = _summary(returns, listing_index, ffr_daily, list(tickers), "inverse_vol")
        invvol_on_best_rows.append({"tickers": tickers, **{k: v for k, v in exact.items() if k != "avg_weights"}})
    pd.DataFrame(invvol_on_best_rows).to_csv(RESULTS_DIR / "exp1_inverse_vol_on_best_equal.csv", index=False)

    iv_results = search_combinations(
        returns, filtered_23, SIMPLE_SEARCH_PORTFOLIO_SIZES, method="inverse_vol", max_combos=max_combos,
    )
    iv_results.to_csv(RESULTS_DIR / "exp1_inverse_vol_search_raw.csv", index=False)
    iv_by_sr = rank_by_sr(iv_results)
    iv_by_sr.head(20).to_csv(RESULTS_DIR / "exp1_inverse_vol_top20_by_SR.csv", index=False)
    top_iv_rows = []
    for _, row in iv_by_sr.head(5).iterrows():
        tickers = tuple(row["tickers"])
        exact = _summary(returns, listing_index, ffr_daily, list(tickers), "inverse_vol")
        top_iv_rows.append({"tickers": tickers, **{k: v for k, v in exact.items() if k != "avg_weights"}})
    top_iv_df = pd.DataFrame(top_iv_rows).sort_values("SR", ascending=False)
    top_iv_df.to_csv(RESULTS_DIR / "exp1_inverse_vol_top_exact.csv", index=False)

    # ---------------------------------------------------------------------
    # 4) Targeted min-vol / risk-parity / max-diversification diagnostics
    #    (Sec. 6.2): duplicate-asset, complementary-asset, extreme-asset (IBIT).
    # ---------------------------------------------------------------------
    print("Running targeted min-vol / risk-parity / max-diversification diagnostics...")
    diagnostic_sets = {
        "duplicate_SPY_AGG": ["SPY", "AGG"],
        "duplicate_SPY_VTV_VUG_AGG": ["SPY", "VTV", "VUG", "AGG"],
        "complementary_SPY_TLT": ["SPY", "TLT"],
        "complementary_SPY_GLD": ["SPY", "GLD"],
        "complementary_SPY_TLT_GLD": ["SPY", "TLT", "GLD"],
        "complementary_SPY_AGG_GLD": ["SPY", "AGG", "GLD"],
        "extreme_SPY_TLT_GLD_IBIT": ["SPY", "TLT", "GLD", "IBIT"],
        "extreme_SPY_AGG_GLD_IBIT": ["SPY", "AGG", "GLD", "IBIT"],
    }
    diag_rows = []
    for name, tickers in diagnostic_sets.items():
        tickers = [tk for tk in tickers if tk in returns.columns]
        for method in ("min_vol", "risk_parity", "max_diversification"):
            summary = _summary(returns, listing_index, ffr_daily, tickers, method)
            diag_rows.append({"test": name, "tickers": tuple(tickers), "method": method, **summary})
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(RESULTS_DIR / "exp1_minvol_riskparity_maxdiv_diagnostics.csv", index=False)

    # ---------------------------------------------------------------------
    # 5) Maximum-diversification broad-universe test (Sec. 6.2): Market,
    #    Sector, Government, Physical and Crypto categories (30 ETFs).
    # ---------------------------------------------------------------------
    print("Running maximum-diversification broad-universe test...")
    broad_categories = ("Market", "Sector", "Government", "Physical", "Crypto")
    broad_universe = [tk for cat in broad_categories for tk in CATEGORIES[cat] if tk in returns.columns]
    log["maxdiv_broad_universe"] = broad_universe
    log["maxdiv_broad_universe_count"] = len(broad_universe)
    broad_summary = _summary(returns, listing_index, ffr_daily, broad_universe, "max_diversification")
    with open(RESULTS_DIR / "exp1_maxdiv_broad_universe.json", "w") as f:
        json.dump({k: v for k, v in broad_summary.items()}, f, indent=2, default=str)

    # ---------------------------------------------------------------------
    # 6) Strongest hindsight-selected simple-strategy portfolio
    # ---------------------------------------------------------------------
    candidates_final = []
    for _, row in top_equal_df.iterrows():
        candidates_final.append({"source": "equal_weight", "tickers": row["tickers"], "SR": row["SR"], "MDD": row["MDD"]})
    for _, row in top_iv_df.iterrows():
        candidates_final.append({"source": "inverse_vol", "tickers": row["tickers"], "SR": row["SR"], "MDD": row["MDD"]})
    for _, row in diag_df.iterrows():
        candidates_final.append({"source": f"diagnostic_{row['method']}", "tickers": row["tickers"], "SR": row["SR"], "MDD": row["MDD"]})
    candidates_final.append({"source": "maxdiv_broad", "tickers": tuple(broad_universe), "SR": broad_summary["SR"], "MDD": broad_summary["MDD"]})
    best = max(candidates_final, key=lambda r: r["SR"])

    log["benchmark_6040"] = {k: v for k, v in benchmark.items() if k != "avg_weights"}
    log["best_spy_agg_gld_split"] = {k: (v if not isinstance(v, dict) else v) for k, v in best_split.items()}
    log["strongest_hindsight_selected_portfolio"] = best
    log["note_hindsight"] = (
        "This portfolio is hindsight-selected: it was identified by searching "
        "over the full 2005-2025 backtest window, as described in Sec. 6.2."
    )
    log["executed_max_combos_per_size"] = max_combos
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    print("Experiment 1 complete. Strongest hindsight-selected portfolio:", best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the paper-scale exhaustive search (every combination)")
    args = parser.parse_args()
    main(full=args.full)
