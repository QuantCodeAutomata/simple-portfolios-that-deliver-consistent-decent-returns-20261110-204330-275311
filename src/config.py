"""Parameters for "Simple Portfolios that Deliver Consistent Decent Returns"
(Wallberg and Karolin, 2026).

Every module-level variable below carries a comment naming its meaning, unit,
and the paper section or table it comes from. No parameter is hidden inside
code; a value changed to fit the replication's time limit keeps both its
paper value and its executed value as separate variables.
"""

# ---------------------------------------------------------------------------
# Table 3: parameters central to the backtest and the trading strategies
# ---------------------------------------------------------------------------
BACKTEST_START = "2005-01-01"          # W_backtest start (Table 3)
BACKTEST_END = "2025-01-01"            # W_backtest end (Table 3)
TRADING_DAYS_PAPER = 5032              # T, trading days over W_backtest (Table 3)
MIN_HISTORY_DAYS = 126                 # H_min, minimum history before investable, days (Table 3)
TARGET_VOL = 0.08                      # sigma_tgt, annual target volatility for vol control (Table 3)
RISK_PARITY_RHO = 1e-5                 # rho, log-barrier regularization in risk parity (Table 3)
TRADING_FREQUENCY = "monthly"          # t_freq (Table 3)
TRADING_COST_KAPPA = 5e-4              # kappa, half-spread trading cost per dollar traded (Table 3, 5bp spread)
ANNUALIZATION_DAYS = 252               # trading days per year used to annualize (paper eq. 28 and surrounding text)

# Markowitz-specific parameters (Table 3)
MARKOWITZ_TURNOVER_LIMIT = 0.10        # T_tgt, turnover cap per rebalance (Table 3)
MARKOWITZ_DELTA_W = 0.05               # delta_w, per-asset position change limit per rebalance (Table 3)
MARKOWITZ_W_MAX = 0.10                 # w_max, maximum weight in a single asset (Table 3)

# ---------------------------------------------------------------------------
# Table 17: risk model parameters (Sec. 6.3)
# ---------------------------------------------------------------------------
COV_LOOKBACK_L = 126                   # L, rolling sample covariance lookback, days (Table 17)
EWMA_HALFLIFE_VAR = 42                 # H_var, EWMA half-life for variances, days (Table 17)
EWMA_HALFLIFE_COV = 126                # H_cov, EWMA half-life for correlations, days (Table 17)
N_STAT_FACTORS = 10                    # k, number of statistical factors (Table 17; first PC ~50%, first 10 ~89% of variance)
FACTOR_EM_MAX_ITER = 200               # EM iteration cap; not specified in the paper, conservative default
FACTOR_EM_TOL = 1e-6                   # EM convergence tolerance on the Frobenius norm of the fitted correlation; not specified in the paper
WINSORIZE_Z = 3.0                      # clip standardized returns at +/-3 std before fitting the factor model; threshold not specified in the paper, conservative default

# Horizon over which realized covariance / returns are evaluated after a monthly forecast
REALIZED_HORIZON_DAYS = 21             # 21 trading days = one month of trading (Sec. 4.1/6.3)

# Sec. 6.3 fits the factor model on "the full historical return matrix available up to
# each forecast date"; this project caps the growing estimation window at ten times the
# correlation half-life (Table 17), beyond which EWMA weights are below 0.1% and
# numerically negligible, to bound per-rebalance computation on the full 59-ETF universe.
FACTOR_MODEL_MAX_LOOKBACK = 1260

# ---------------------------------------------------------------------------
# Table 19/20: alpha model parameters (Sec. 5.2, 5.3, 6.3)
# ---------------------------------------------------------------------------
RIDGE_LOOKBACK_H = 252                 # H_ridge, rolling ridge training window, days (Table 19)
RIDGE_HALFLIFE = 252                   # half-life of the training-sample weights, days (Table 19)
RIDGE_FORECAST_HORIZON = 21            # h, forecast horizon, days (Table 19)
MOMENTUM_LOOKBACK = 21                 # L_mom, one-month momentum lookback, days (Table 20)
MOVAVG_LOOKBACK = 50                   # L_mov, moving-average lookback, days (Table 20)
MOVAVG_RHO = 1.0                       # rho, moving-average threshold multiplier (Table 20)

# Non-overlapping feature windows for ridge regression (Sec. 5.3):
#   STR_t = -log(p_{t-1}/p_{t-21})
#   MTM_t = log(p_{t-21}/p_{t-252})
#   LTR_t = -log(p_{t-252}/p_{t-756})
STR_WINDOW = (1, 21)
MTM_WINDOW = (21, 252)
LTR_WINDOW = (252, 756)
RIDGE_LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]  # sweep of the ridge penalty lambda (paper Fig. 25 sweeps "several orders of magnitude"; exact grid not specified)

# ---------------------------------------------------------------------------
# Experiment 1: exhaustive and targeted simple-strategy search (Sec. 6.2)
# ---------------------------------------------------------------------------
SIMPLE_SEARCH_PORTFOLIO_SIZES = (3, 4, 5)   # enumerate 3-, 4-, and 5-ETF combinations (Sec. 6.2)
EXCLUDED_CATEGORIES_FOR_SEARCH = ("Country", "Futures")  # excluded from the 23-ETF filtered search universe (Sec. 6.2)
FIXED_SPLIT_GRID_STEP = 0.10            # 10 percentage-point grid for SPY/AGG/GLD split enumeration (Sec. 6.2)
BENCHMARK_6040 = {"SPY": 0.6, "AGG": 0.4}  # 60/40 benchmark fixed weights

# IBIT bitcoin backfill (Sec. 6.2); the paper does not specify the exact source or
# convention, so IBIT's pre-listing history is reconstructed from Dataloader crypto
# bars (ticker "X:BTCUSD") starting 2017-01-01, an assumption recorded in
# results/RESULTS.md, not a paper fact.
IBIT_BACKFILL_START = "2017-01-01"
IBIT_BACKFILL_SOURCE_TICKER = "X:BTCUSD"

# ---------------------------------------------------------------------------
# Experiment 3: category ablation (Sec. 6.3, Table 22/23/24)
# ---------------------------------------------------------------------------
# Categories removed to build the paper's "Momentum Universe": the paper does not print
# an explicit removal list, but states the resulting universe holds 26 ETFs across six
# categories. Removing {Sector, Country, Futures} (11 + 15 + 7 = 33 of 59 ETFs) leaves
# exactly 26 ETFs in the remaining six categories, matching Sec. 6.3's statement and the
# three categories whose individual removal in Table 22 raises the Sharpe ratio the most
# (Futures 0.53, Countries 0.51, Sectors 0.47, all above the 0.43 baseline).
MOMENTUM_UNIVERSE_REMOVED_CATEGORIES = ("Sector", "Country", "Futures")
# For the moving-average signal, Sec. 6.3 states in prose that removing Countries,
# Government, or Global improves the portfolio while removing Market or Sectors
# makes it worse (Table 23); these three categories are removed for its post-ablation run.
MOVAVG_UNIVERSE_REMOVED_CATEGORIES = ("Country", "Government", "Global")

# ---------------------------------------------------------------------------
# Experiment 4: proposed portfolios and truncated-window robustness (Sec. 6.4, Table 25/30)
# ---------------------------------------------------------------------------
TRUNCATED_WINDOW_EXCL_FIRST5 = ("2010-01-01", "2025-01-01")  # excludes the Financial Crisis
TRUNCATED_WINDOW_EXCL_LAST5 = ("2005-01-01", "2020-01-01")   # excludes COVID-19 / start of the Russia-Ukraine war
REGRET_BOUND_TRADING_FREQUENCY = "yearly"  # the regret-bound upper-bound strategy trades only yearly (Sec. 3.3/6.4)

# ---------------------------------------------------------------------------
# Execution-time scaling (not a paper parameter; see results/RESULTS.md)
# ---------------------------------------------------------------------------
# The paper's exhaustive search enumerates every 3-5 ETF combination among the 23-ETF
# (resp. 59-ETF) universe. The --full flag in exp/exp1.py selects these paper-scale
# counts; otherwise a time-bounded subsample is used (see RESULTS.md for the measured
# runtime and the narrowing rule applied).
EXHAUSTIVE_SEARCH_MAX_COMBOS_RUN = 4000   # executed cap per portfolio size when --full is not given
