# Simple Portfolios that Deliver Consistent Decent Returns

Reproduction of selected experiments from Wallberg and Karolin (2026),
"Simple Portfolios that Deliver Consistent Decent Returns".

This repository implements the paper's daily backtest engine (Sec. 6.1), the
six simple portfolio-construction strategies of Sec. 3.2, the three
covariance-forecasting methods of Sec. 4, the alpha models of Sec. 5, the
modified long-only Markowitz allocation of Sec. 3.3, and the four selected
experiments of Sec. 6:

* `exp.exp1` -- simple long-only strategies on small ETF subsets (Sec. 6.2).
* `exp.exp2` -- covariance-forecast accuracy (Sec. 6.3, Table 18).
* `exp.exp3` -- alpha models and category-ablated Markowitz portfolios
  (Sec. 6.3, Tables 19-24).
* `exp.exp4` -- final proposed portfolios and truncated-window robustness
  (Sec. 6.4, Table 25).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

Daily OHLCV bars are loaded through the QuantX Dataloader REST API
(`data/data_loader.py`); the daily effective federal funds rate (the paper's
cash leg) is loaded from Massive's funding-conditions endpoint, which
republishes the FRED DFF series used in the paper. No market data is stored
in this repository; `data/cache/` holds only a local, git-ignored cache of
API responses.

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic fixtures
python -m pytest                         # full suite, needs API keys
```

## Experiments

```bash
python -m exp.exp1   # add --full for the paper-scale exhaustive search
python -m exp.exp2
python -m exp.exp3    # add --full for daily (vs. monthly) ridge refitting
python -m exp.exp4
```

Every run overwrites the tables, metrics and plots in `results/`. See
`results/RESULTS.md` for the comparison against the paper's reported values,
deviations, and what was not reproduced.

## Layout

```
src/            backtest engine, strategies, risk models, alpha, Markowitz, metrics, config
exp/            experiment scripts (exp1..exp4)
data/           data_loader.py (QuantX Dataloader + Massive FFR)
tests/          pytest suite
results/        generated tables, metrics, plots, RESULTS.md
```
