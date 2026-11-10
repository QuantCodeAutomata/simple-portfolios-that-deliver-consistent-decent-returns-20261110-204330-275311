"""Fast unit tests for exp/exp4.py helpers on small synthetic data.

No network calls: these exercise ``yearly_rebalance_dates``, ``summarize``
and the portfolio builders directly on fabricated return panels wrapped in
real ``UniversePanel`` instances, not the real market-data pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestResult, run_backtest
from src.market_data import UniversePanel
from src.portfolio_runner import listing_day_index

from exp.exp4 import (
    ALL_BUILDERS,
    IMPLEMENTABLE_BUILDERS,
    REGRET_BOUND_LABEL,
    REGRET_BOUND_NAME,
    build_6040,
    build_equal_spy_agg_gld,
    build_maxdiv_spy_tlt_gld_ibit,
    build_momentum_universe_markowitz,
    build_regret_bound,
    build_spy_vc8,
    label_of,
    summarize,
    yearly_rebalance_dates,
)


def _synthetic_panel(
    tickers: list[str], n_days: int, seed: int = 0, start: str = "2010-01-01",
) -> UniversePanel:
    """A small synthetic UniversePanel with iid Gaussian daily returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    data = rng.normal(loc=0.0003, scale=0.01, size=(n_days, len(tickers)))
    returns = pd.DataFrame(data, index=dates, columns=tickers)
    prices = 100.0 * (1.0 + returns).cumprod()
    listing_index = listing_day_index(returns)
    ffr_daily = pd.Series(0.00005, index=dates)
    return UniversePanel(
        prices=prices, returns=returns, listing_index=listing_index, ffr_daily=ffr_daily, meta={},
    )


def _constant_weight_strategy(weights: dict):
    def fn(date, t, w_pre):
        return dict(weights)
    return fn


class TestYearlyRebalanceDates:
    def test_strictly_increasing_and_about_one_per_year(self):
        dates = pd.bdate_range("2015-01-01", "2022-12-31")
        rebal = yearly_rebalance_dates(dates)

        assert len(rebal) > 0
        assert (rebal.to_series().diff().dropna() > pd.Timedelta(0)).all()
        n_years = dates.year.nunique()
        # one rebalance date per calendar year present in the index
        assert len(rebal) == n_years

    def test_each_date_is_the_last_trading_day_of_its_year(self):
        dates = pd.bdate_range("2019-01-01", "2021-12-31")
        rebal = yearly_rebalance_dates(dates)

        for d in rebal:
            same_year = dates[dates.year == d.year]
            assert d == same_year.max()

    def test_single_year_returns_one_date(self):
        dates = pd.bdate_range("2020-01-01", "2020-06-30")
        rebal = yearly_rebalance_dates(dates)
        assert len(rebal) == 1
        assert rebal[0] == dates.max()


class TestSummarize:
    def test_keys_and_ranges(self):
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2020-01-01", periods=300)
        returns = pd.DataFrame(
            rng.normal(0.0003, 0.01, size=(300, 3)), index=dates, columns=["A", "B", "C"],
        )
        ffr = pd.Series(0.00005, index=dates)
        strategy = _constant_weight_strategy({"A": 0.5, "B": 0.3})
        result = run_backtest(returns, strategy, ffr)

        out = summarize(result)

        assert set(out.keys()) == {"r_e", "sigma", "MDD", "TO", "SR", "LQ"}
        assert out["sigma"] >= 0.0
        assert 0.0 <= out["MDD"] <= 1.0
        for v in out.values():
            assert isinstance(v, float)


class TestLabelOf:
    def test_regret_bound_is_flagged_not_implementable(self):
        label = label_of(REGRET_BOUND_NAME)
        assert label == REGRET_BOUND_LABEL
        assert "NOT IMPLEMENTABLE" in label

    def test_other_names_pass_through_unchanged(self):
        assert label_of("60_40_benchmark") == "60_40_benchmark"


class TestFixedWeightBuilders:
    def test_build_6040_invests_only_in_spy_agg(self):
        panel = _synthetic_panel(["SPY", "AGG", "GLD"], n_days=400, seed=2)
        result = build_6040(panel, momentum_universe=[])

        assert isinstance(result, BacktestResult)
        assert (result.nav > 0).all()
        assert (result.weights["GLD"] == 0.0).all()
        # SPY should carry roughly 60% and AGG roughly 40% whenever invested
        invested = result.weights[(result.weights["SPY"] + result.weights["AGG"]) > 1e-6]
        if len(invested):
            ratio = (invested["SPY"] / (invested["SPY"] + invested["AGG"])).mean()
            assert ratio == pytest.approx(0.6, abs=0.05)

    def test_build_spy_vc8_scales_down_a_high_vol_series(self):
        # Make SPY's synthetic volatility much higher than the 8% annual
        # target so volatility control must scale weights below 1.
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2010-01-01", periods=400)
        returns = pd.DataFrame({"SPY": rng.normal(0.0, 0.03, size=400)}, index=dates)
        prices = 100.0 * (1.0 + returns).cumprod()
        listing_index = listing_day_index(returns)
        ffr_daily = pd.Series(0.0, index=dates)
        panel = UniversePanel(prices=prices, returns=returns, listing_index=listing_index,
                               ffr_daily=ffr_daily, meta={})

        result = build_spy_vc8(panel, momentum_universe=[])

        nonzero = result.weights.loc[result.weights["SPY"] > 0, "SPY"]
        assert len(nonzero) > 0
        assert nonzero.max() < 1.0

    def test_build_equal_spy_agg_gld_is_roughly_one_third_each(self):
        panel = _synthetic_panel(["SPY", "AGG", "GLD"], n_days=400, seed=4)
        result = build_equal_spy_agg_gld(panel, momentum_universe=[])

        invested = result.weights[result.weights.sum(axis=1) > 1e-6]
        assert len(invested) > 0
        for tk in ("SPY", "AGG", "GLD"):
            assert invested[tk].mean() == pytest.approx(1.0 / 3, abs=0.02)

    def test_build_maxdiv_runs_on_four_assets(self):
        panel = _synthetic_panel(["SPY", "TLT", "GLD", "IBIT"], n_days=400, seed=5)
        result = build_maxdiv_spy_tlt_gld_ibit(panel, momentum_universe=[])

        assert isinstance(result, BacktestResult)
        assert (result.nav > 0).all()
        assert (result.weights.sum(axis=1) <= 1.0 + 1e-6).all()


class TestMarkowitzBuilders:
    def test_momentum_universe_markowitz_runs_on_tiny_universe(self):
        tickers = ["A", "B", "C", "D", "E"]
        panel = _synthetic_panel(tickers, n_days=500, seed=6)

        result = build_momentum_universe_markowitz(panel, momentum_universe=tickers)

        assert isinstance(result, BacktestResult)
        assert (result.nav > 0).all()
        assert (result.weights.to_numpy() >= -1e-9).all()
        assert (result.weights.sum(axis=1) <= 1.0 + 1e-6).all()

    def test_regret_bound_uses_yearly_rebalance_dates(self):
        tickers = ["A", "B", "C"]
        panel = _synthetic_panel(tickers, n_days=600, seed=7)

        result = build_regret_bound(panel, momentum_universe=[])

        assert isinstance(result, BacktestResult)
        assert (result.nav > 0).all()
        expected = yearly_rebalance_dates(panel.returns.index)
        assert list(result.rebalance_dates) == list(expected)


class TestBuilderRegistries:
    def test_implementable_builders_exclude_regret_bound(self):
        assert REGRET_BOUND_NAME not in IMPLEMENTABLE_BUILDERS
        assert len(IMPLEMENTABLE_BUILDERS) == 5

    def test_all_builders_include_regret_bound(self):
        assert REGRET_BOUND_NAME in ALL_BUILDERS
        assert len(ALL_BUILDERS) == 6
        assert set(ALL_BUILDERS) == set(IMPLEMENTABLE_BUILDERS) | {REGRET_BOUND_NAME}
