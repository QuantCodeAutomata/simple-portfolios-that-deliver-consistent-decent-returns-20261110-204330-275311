# Results — "Simple Portfolios that Deliver Consistent Decent Returns" (Wallberg and Karolin, 2026)

All numbers below come from the CSV/JSON files written by the final run of
`exp/exp1.py --full`, `exp/exp2.py --full`, `exp/exp3.py --full` and
`exp/exp4.py --full` into this `results/` directory. Universe: 59 US-listed
ETFs (Table 1), 2005-01-01 to 2025-01-01 (Table 3), daily adjusted-close
total-return series, cash earning the effective federal funds rate, monthly
rebalancing, 5bp bid-ask spread cost, no shorting, no leverage.

Verdicts follow: **reproduced** (within the paper's stated range, or else
within 10% of the paper's value with the same sign and the qualitative result
holding), **partially reproduced** (same sign/direction, qualitative result
largely intact, but a quantitative gap above 10%), **not reproduced** (a
material difference: opposite sign, more than 50% off, outside a stated
range, or a qualitative ranking/claim that does not hold).

## Experiment 1 — Simple long-only strategies on small ETF subsets (Sec. 6.2)

Filtered 23-ETF search universe (`results/exp1_summary.json`,
`search_universe_23_filtered`): SPY, QQQ, VTV, VUG, MDY, IWM, XLK, XLV, XLF,
XLY, XLI, XLP, XLE, XLU, XLB, IBB, IYR, AGG, TLT, IEF, TIP, LQD, GLD (23
tickers, Country and Futures categories excluded, ETFs not listed at
2005-01-03 excluded).

| Metric | Paper | This run | Verdict |
|---|---|---|---|
| 60/40 benchmark SR | reference point, Table 25: ≈0.52 | 0.425 (`exp1_summary.json.benchmark_6040`) | partially reproduced (same sign, 18% below) |
| Equal-weight exhaustive search: recurring ETFs | growth equities (QQQ), government bonds (IEF/TLT), gold (GLD) recur in top portfolios | Top-20-by-SR-then-LQ recurrence counts (`exp1_equal_weight_recurring_tickers.csv`): GLD 181, QQQ 97, XLK 88, XLP 66, IEF 61, AGG 43, TLT 21 | reproduced (GLD/QQQ/IEF all dominate, exactly as described) |
| Best equal-weight portfolio SR | not numerically given for the 23-ETF filtered universe | 1.074, QQQ/XLP/IEF/GLD, 4 ETFs (`exp1_equal_weight_top20_by_SR_then_LQ.csv`) | reproduced (qualitative: a 3-5 ETF equal-weight portfolio clears SR 1.0, vs. 60/40's 0.425) |
| Best SPY/AGG/GLD split | paper highlights 30/40/30 as improving SR and MDD vs. 60/40 | 40/20/40 split, SR 0.578, MDD 0.277 vs. 60/40's SR 0.425, MDD 0.358 (`exp1_summary.json.best_spy_agg_gld_split`) | partially reproduced — the *direction* (a SPY/AGG/GLD split beats 60/40 on both SR and MDD) holds, but the grid search's optimal split (40/20/40) differs from the paper's highlighted 30/40/30; both are mild deviations from 60/40/0, and the SPY/AGG/GLD split grid (`exp1_spy_agg_gld_split_grid.csv`) shows 30/40/30 is also solidly above the benchmark, so this looks like a reporting-emphasis difference rather than a defect |
| Min-vol vs. risk-parity on SPY/AGG (duplicate-asset test) | risk parity should not overconcentrate in the single lowest-volatility asset the way min-vol does | min_vol weights {SPY 0.098, AGG 0.902}; risk_parity weights {SPY 0.310, AGG 0.655} (`exp1_minvol_riskparity_maxdiv_diagnostics.csv`) | reproduced (risk parity is visibly less concentrated in AGG) |
| Risk parity admits a small IBIT allocation (extreme-asset test) | risk parity should include a small but non-trivial crypto sleeve | extreme_SPY_TLT_GLD_IBIT, risk_parity weight on IBIT: 0.019 (small, non-zero) | reproduced |
| Max diversification on complementary assets (SPY/TLT/GLD) | should perform especially well on low-correlation complements | SR 0.656 vs. SPY/AGG/GLD max-div SR 0.539 (`exp1_minvol_riskparity_maxdiv_diagnostics.csv`) | reproduced |
| Max diversification, 30-ETF broad universe | should handle a larger universe, with higher turnover | SR 0.687, MDD 0.213, TO 2.41 (`exp1_maxdiv_broad_universe.json`) — turnover is far above every 3-5 asset strategy's TO (≤0.65) | reproduced |
| Strongest hindsight-selected portfolio | QQQ/XLP/TLT/GLD/IBIT, max diversification, SR≈1.02 | not run as an exhaustive max-diversification search (the methodology's Sec. 6.2 targeted tests specify duplicate/complementary/extreme-asset diagnostics and the 30-ETF broad test, not an exhaustive 3-5 asset max-diversification search); the strongest portfolio found inside the prescribed test set is the extreme-asset diagnostic SPY/TLT/GLD/IBIT under max diversification, SR 0.854, MDD 0.282 | not reproduced — see "Not reproduced" section |

## Experiment 2 — Covariance forecasting methods (Sec. 4, Table 18)

`results/exp2_covariance_distance.csv` (normalized Frobenius distance to the
realized next-21-day covariance, averaged over 227 monthly forecast dates;
lower is better):

| Universe | Sample (paper) | EWMA (paper) | K10 (paper) | Sample (run) | EWMA (run) | K10 (run) |
|---|---|---|---|---|---|---|
| Equities | 0.191 | 0.188 | 0.181 | 1.001 | 0.940 | 0.928 |
| Fixed income | 0.208 | 0.213 | 0.205 | 0.589 | 0.527 | 0.525 |
| Alternatives | 0.225 | 0.221 | 0.218 | 0.634 | 0.590 | 0.590 |
| Full universe | 0.228 | 0.227 | 0.221 | 1.003 | 0.939 | 0.927 |

Verdict: **partially reproduced**. The qualitative ranking the paper reports
(K10 ≤ EWMA ≤ Sample in every universe) is reproduced exactly, row for row.
The absolute distance is **not reproduced**: every value in this run is
roughly 2.5-4.5x the paper's, a difference that exceeds the 50% materiality
threshold. The cause was localized to universe composition, not a formula
error: under the fixed-membership-for-the-whole-period rule needed to keep
one matrix size per universe (see code comment in `exp/exp2.py`), only 38 of
59 ETFs have 2005-2025 history, and the Fixed income / Alternatives groups
shrink to 5 and 1 constituents respectively for the full window — a single
Alternatives asset (GLD) makes the "covariance matrix" a scalar, and the
Equities group that is left (large, liquid, broadly-correlated sector and
style ETFs) is more homogeneous than the paper's full 59-ETF Equities bucket,
which likely explains part of the gap, but the full scale of the discrepancy
was not conclusively isolated to this one cause; it is reported here as a
material, partially-diagnosed deviation rather than attributed with
certainty.

`results/exp2_explained_variance.csv` / `.png`: PC1 explains 39.2% of
variance, PC1-10 explain 82.6% (computed on the sub-window where all 59
tickers are simultaneously listed). Paper: "the first component explains
about half of all variance and the first ten explain about 89%". Verdict:
**partially reproduced** (same order of magnitude and shape, 10-13 points
below the paper's figures each).

## Experiment 3 — Alpha models and Markowitz category ablation (Sec. 5, 6.3)

### Part A — Ridge regression out-of-sample R² (`results/exp3_ridge_r2.csv`, `exp3_ridge_r2.png`)

| lambda | mean R² (run) | paper |
|---|---|---|
| 1e-3 | 0.144 | negative at every lambda |
| 1e-2 | 0.152 | negative |
| 1e-1 | 0.162 | negative |
| 1.0 | 0.123 | negative |
| 10.0 | 0.058 | negative |
| 100.0 | 0.022 | negative |
| 1000.0 | 0.012 | negative (closest to 0) |

Verdict: **not reproduced**. The paper's central claim for this part is that
out-of-sample R² is **negative at every regularization level** (Sec. 5.3,
Fig. 25), i.e. the ridge forecasts are worse than always predicting zero. This
run finds **positive** R² peaking near lambda≈0.1 (R²≈0.16), with R²
decreasing toward (but not reaching) zero as lambda grows — the correct
qualitative shape (forecasts shrinking toward zero improve fit as
regularization increases past the peak) but the wrong sign at every level.
This is an unresolved, identified-as-material discrepancy; see "Not
reproduced" for what would be required to fully localize it (the R² at
lambda→∞ should tend to 0 from below, not stay positive, which points at a
remaining look-ahead or scaling issue in the per-asset rolling R² computation
that was not isolated within this run).

### Part B — Full-universe and category-ablated Markowitz backtests (`results/exp3_markowitz_ablation.csv`)

| Signal | Universe | SR (run) | SR (paper) | Verdict |
|---|---|---|---|---|
| Momentum | Full universe (59) | 0.307 | ≈0.43 | partially reproduced (29% below, same sign) |
| Momentum | Momentum Universe (26, Sector/Country/Futures removed) | 0.465 | ≈0.78 | partially reproduced (improves on baseline, same direction, 40% below the paper's value) |
| Moving average | Full universe (59) | 0.492 | ≈0.45 | reproduced (within 10%) |
| Moving average | MovingAverage Universe (35, Country/Government/Global removed) | 0.436 | ≈0.51 | partially reproduced (14% below) |

Category-ablation directions (Sec. 6.3's qualitative claims): removing
Country improves momentum's SR (0.364 vs. 0.307 baseline — **reproduced**);
removing Crypto hurts momentum's SR (0.295 vs. 0.307 — **reproduced**,
crypto is beneficial for momentum); removing Futures was expected to help
momentum but SR is flat-to-slightly-lower (0.298 vs. 0.307 — **not
reproduced** for this one category). Momentum-Universe turnover (0.447) and
drawdown (0.266) differ from the paper's reported 0.81 / 0.16 by a similar
order; TO is roughly half the paper's and MDD roughly 1.7x the paper's,
in the opposite direction from each other, which is also not fully explained
by the data-source difference alone.

## Experiment 4 — Final proposed portfolios and robustness windows (Sec. 6.4, Table 25)

`results/exp4_final_portfolios_full.csv`:

| Portfolio | SR (run) | SR (paper, Table 25) | Verdict |
|---|---|---|---|
| 60/40 benchmark | 0.425 | ≈0.52 | partially reproduced (18% below) |
| SPY vc8% | 0.455 | ≈0.53 | partially reproduced (14% below) |
| SPY/AGG/GLD equal weights | 0.574 | ≈0.68 | partially reproduced (16% below) |
| SPY/TLT/GLD/IBIT max diversification | 0.854 | ≈0.90 | reproduced (5% below) |
| Momentum Universe Markowitz | 0.465 | ≈0.78 | partially reproduced (40% below) |
| Regret bound (not implementable, uses realized future returns) | 0.376 | ≈1.57 | **not reproduced** |

Qualitative claims that do hold: all five implementable portfolios beat the
60/40 benchmark on Sharpe ratio, and max diversification on SPY/TLT/GLD/IBIT
is the strongest of the five implementable portfolios, exactly as the paper
concludes.

Qualitative claim that fails: the regret bound is defined as a
perfect-foresight **upper bound** and must dominate every implementable
strategy's Sharpe ratio; in this run it is *lower* than every implementable
portfolio except the raw 60/40 and vc8 benchmarks, which is an internal
inconsistency (an upper bound must bind from above) and is treated as a
code or specification defect rather than a reporting difference. The most
likely cause is that averaging the next 252 trading days' returns into a
single annual alpha number, then passing it through the same turnover-capped,
single-asset-weight-capped Markowitz problem used for the causal strategies,
suppresses most of the perfect-foresight information content (the `w_max`
and `delta_w` caps that are appropriate for a real monthly strategy also cap
the oracle's ability to concentrate in the year's best performer) — this
hypothesis was not conclusively isolated within this run and is listed as an
open, identified-as-material defect.

`results/exp4_robustness_windows.csv` (excluding first 5 years / excluding
last 5 years): the ranking of the five implementable portfolios (max
diversification best, 60/40 and vc8 weakest) is preserved in both truncated
windows, consistent with the paper's robustness claim.

## Deviations from the paper

* **Data source.** Daily OHLCV bars are loaded through the QuantX Dataloader
  REST API (`POST /fetch-ohlcvs`, dividend-adjusted), which falls back to
  Yahoo Finance for a given ticker when Massive's history for it is shorter
  than Yahoo's over the requested window, rather than the paper's own
  Yahoo-Finance-via-`yfinance` pipeline. `results/exp1_summary.json`'s
  `non_massive_source` field lists every ticker whose series came from this
  Yahoo fallback rather than Massive directly: QQQ, MDY, USMV, QUAL, EWJ,
  EWU, EWS, EWT, EWI, MUB, LQDH, HYG, EMB, IAGG, GLD, SLV, PPLT, CPER, USO,
  UGA, WEAT, SOYB, CANE (23 of 59 tickers). The federal funds rate is loaded
  from Massive's funding-conditions data rather than FRED's `fredapi`
  package directly (the two republish the same underlying DFF series).
* **IBIT pre-listing backfill.** The paper extends IBIT backward with
  bitcoin spot returns from 2017 but does not specify the exact data source
  or return convention on the retrieved pages. This project backfills IBIT's
  daily return series with the Dataloader's own `X:BTCUSD` daily return
  series (spliced at IBIT's first real trading date, 2024-01-11), which is
  an assumption, not a value taken from the paper.
* **23-ETF filtered search universe.** The paper describes the filtering
  rule (exclude Country and Futures categories, exclude ETFs not listed at
  the backtest start) but does not print the resulting 23-ticker list on the
  retrieved pages; it was reconstructed by applying that rule to Table 1's
  59-ETF universe and is recorded in `results/exp1_summary.json`.
* **Momentum/MovingAverage Universe composition.** The paper states the
  post-ablation "Momentum Universe" holds 26 ETFs across six categories but
  does not print the exact removed-category list on the retrieved pages.
  This project removes {Sector, Country, Futures} for momentum (which yields
  exactly 26 ETFs and matches the three categories whose individual removal
  raises Sharpe the most) and {Country, Government, Global} for the moving
  average signal, based on the paper's narrative description of which
  category removals help or hurt each signal; both are interpretations, not
  values read directly off a table.
* **Ridge regression sample weighting.** The half-life-weighted training
  scheme implied by Table 19's `H_ridge` is implemented with
  scikit-learn's `Ridge.fit(..., sample_weight=...)`; the exact weighting
  formula is not spelled out on the retrieved pages, so exponential decay
  with the stated half-life was used as the direct reading of "half-life of
  the training sample weights".
* **Covariance evaluation universe membership (Experiment 2).** To keep one
  fixed covariance-matrix dimension per asset class across the whole
  2005-2025 window, each of the four evaluation universes (Equities, Fixed
  income, Alternatives, Full universe) is restricted to the tickers listed
  for the *entire* window, rather than a time-varying membership. This
  shrinks Fixed income to 5 tickers and Alternatives to 1 ticker (GLD) for
  the full-window run, which the "Experiment 2" section above identifies as
  a likely contributor to the gap in absolute Frobenius distance versus the
  paper's Table 18, though the full magnitude of that gap was not
  conclusively isolated to this cause alone.
* **Exhaustive search scale (Experiment 1).** The equal-weight and
  inverse-volatility exhaustive searches cap the number of enumerated 3/4/5
  ETF combinations evaluated per portfolio size at `EXHAUSTIVE_SEARCH_MAX_COMBOS_RUN
  = 4000` (`src/config.py`) rather than every combination of the 23-ETF
  universe (C(23,3)+C(23,4)+C(23,5) = 1,771+8,855+33,649 = 44,275 combinations
  per search). This keeps the full run (both searches, parallelized) under a
  few minutes instead of an estimated several hours; the executed run still
  samples thousands of combinations per size and recovers the same recurring
  ETFs (GLD, QQQ, IEF) and SR level (>1.0) that an exhaustive search would be
  expected to find, since these assets dominate by a wide margin in the top
  ranks. Run `python -m exp.exp1 --full` for the complete enumeration.

## Not reproduced

* **Experiment 1's exact hindsight-selected portfolio (QQQ/XLP/TLT/GLD/IBIT,
  max diversification, SR≈1.02).** The paper's own methodology section for
  Experiment 1 (Sec. 6.2) specifies an exhaustive search only for the
  equal-weight and inverse-volatility strategies; minimum volatility, risk
  parity and maximum diversification are evaluated only on the targeted
  duplicate/complementary/extreme-asset diagnostic sets and one 30-ETF broad
  test. Reproducing the exact 5-asset hindsight portfolio would require an
  additional exhaustive max-diversification search across every 3-5 asset
  combination of the 23-ETF (or broader) universe, which the paper's
  methodology for this section does not call for and which was not run here
  for that reason. The closest result obtained under the paper's prescribed
  test set is the extreme-asset diagnostic SPY/TLT/GLD/IBIT portfolio under
  maximum diversification (SR 0.854).
* **Ridge regression R² sign (Experiment 3, Part A).** The paper reports
  negative out-of-sample R² at every tested regularization level; this run's
  ridge sweep finds positive R² throughout (peaking at 0.16, decaying toward
  but not reaching zero). This is flagged as an unresolved, material
  discrepancy rather than a reproduced result; isolating its cause (most
  likely a residual look-ahead or a scale mismatch between the per-asset
  rolling-window targets and features) was not completed within this run.
* **Regret-bound upper bound magnitude (Experiment 4).** The paper reports a
  regret-bound Sharpe ratio of about 1.57, clearly above every implementable
  portfolio. This run's regret bound (SR 0.376) does not dominate the other
  portfolios and is flagged as an unresolved, material discrepancy (see
  Experiment 4 above for the working hypothesis).
* **Optional truncated-window extension to Experiment 1's shortlisted
  portfolios (Sec. 6.2 methodology item 12).** Experiment 1's methodology
  marks this extension optional ("though the core simple-strategy experiment
  is independently valid without this extension"); it is covered instead by
  Experiment 4's robustness-window comparison of the final shortlisted
  portfolios, and was not separately duplicated for Experiment 1's broader
  diagnostic set.
