# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this implements

A reproduction of selected experiments from Wallberg and Karolin (2026),
"Simple Portfolios that Deliver Consistent Decent Returns": the daily
backtest engine and six simple portfolio-construction strategies (Sec. 3.2,
6.1-6.2), the three covariance-forecasting methods (Sec. 4, 6.3), the alpha
models and modified long-only Markowitz allocator with category ablations
(Sec. 3.3, 5, 6.3), and the final proposed portfolios with truncated-window
robustness checks (Sec. 6.4). See `README.md` for the per-experiment mapping
and `results/RESULTS.md` for the numeric comparison against the paper.

## Layout

```
src/     flat library modules (backtest engine, strategies, risk models,
         alpha signals, Markowitz solver, metrics, universe, config.py)
exp/     experiment scripts, one per experiment (exp1..exp4), runnable as
         `python -m exp.<name>`
data/    data_loader.py: OHLCV bars via the QuantX Dataloader REST API and
         the federal funds rate via Massive; no downloaded data is committed
tests/   pytest suite (unit tests on synthetic fixtures; market-data tests
         marked `market_data`)
results/ generated tables, metrics and plots, plus RESULTS.md
```

No other top-level files or directories are used, besides `README.md`,
`requirements.txt`, `.gitignore`, `.env.example` and this file.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no API keys needed
python -m pytest                         # full suite, needs API keys

python -m exp.exp1   # add --full for the paper-scale exhaustive search
python -m exp.exp2
python -m exp.exp3   # add --full for daily (vs. monthly) ridge refitting
python -m exp.exp4
```

## Conventions

* All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables with a comment naming their paper section or stating
  that the paper does not specify them; no parameter is hardcoded elsewhere.
* Market data is loaded only through `data/data_loader.py` (QuantX Dataloader
  for OHLCV bars, Massive for the federal funds rate); `src/` and `exp/`
  never call a data provider directly.
* API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from an
  uncommitted `.env` file; only empty placeholders live in `.env.example`.
* Running an experiment overwrites its files in `results/`. The numbers and
  verdicts in `results/RESULTS.md` are not regenerated automatically — after
  a rerun, check them against the new result files and update the Markdown
  by hand.
* Unit tests run on small fixtures or synthetic data and must pass without
  API keys (`python -m pytest -m "not market_data"`). Tests that load real
  market data are marked `market_data`, use a reduced, session-cached slice
  of the universe, and are skipped automatically when `DATALOADER_API_KEY`
  is unset.
