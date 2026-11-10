"""Fast unit tests for exp/exp3.py helpers on small synthetic data.

No network calls: these exercise ``summarize``, ``ridge_r2_sweep`` and
``restrict_panel`` directly on fabricated inputs, not the real market-data
pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.config import RIDGE_FORECAST_HORIZON, RIDGE_LOOKBACK_H
from src.portfolio_runner import listing_day_index
from exp.exp3 import RIDGE_MIN_ROWS, restrict_panel, ridge_r2_sweep, summarize


def _synthetic_returns_frame(n_days=300, n_assets=4, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_assets)]
    data = rng.normal(loc=0.0003, scale=0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _constant_weight_strategy(weights: dict):
    def fn(date, t, w_pre):
        return dict(weights)
    return fn


class TestSummarize:
    def test_keys_and_ranges(self):
        returns = _synthetic_returns_frame()
        ffr = pd.Series(0.00005, index=returns.index)
        strategy = _constant_weight_strategy({"T0": 0.5, "T1": 0.3})
        result = run_backtest(returns, strategy, ffr)

        out = summarize(result)

        assert set(out.keys()) == {"r_e", "sigma", "MDD", "TO", "SR", "LQ"}
        assert out["sigma"] >= 0.0
        assert 0.0 <= out["MDD"] <= 1.0
        for v in out.values():
            assert isinstance(v, float)

    def test_zero_turnover_for_buy_and_hold_like_constant_target(self):
        # A strategy that always targets the same weights still has nonzero
        # turnover unless the portfolio already sits exactly there (drift),
        # but turnover must never be negative and NAV must stay positive.
        returns = _synthetic_returns_frame(n_days=120, n_assets=3, seed=1)
        ffr = pd.Series(0.0, index=returns.index)
        strategy = _constant_weight_strategy({"T0": 1.0 / 3, "T1": 1.0 / 3, "T2": 1.0 / 3})
        result = run_backtest(returns, strategy, ffr)

        assert (result.turnover >= -1e-12).all()
        assert (result.nav > 0).all()
        out = summarize(result)
        assert np.isfinite(out["TO"])


class TestRidgeR2Sweep:
    def test_aggregation_columns_and_shape(self, monkeypatch):
        """Fabricate rolling_ridge_oos_r2 so the sweep/aggregation logic is
        tested without running real ridge regressions."""
        import exp.exp3 as exp3

        def fake_r2(series, lam, **kwargs):
            # Deterministic function of (series length, lambda) so we can
            # check the aggregation arithmetic exactly.
            return 1.0 / lam

        monkeypatch.setattr(exp3, "rolling_ridge_oos_r2", fake_r2)

        n_rows = RIDGE_MIN_ROWS + 10
        dates = pd.bdate_range("2000-01-01", periods=n_rows)
        prices = pd.DataFrame(
            {
                "A": np.linspace(100, 110, n_rows),
                "B": np.linspace(50, 55, n_rows),
                "SHORT": np.concatenate([np.full(n_rows - 5, np.nan), np.linspace(10, 11, 5)]),
            },
            index=dates,
        )
        lambdas = [1.0, 2.0, 4.0]

        per_asset, agg, skipped = ridge_r2_sweep(["A", "B", "SHORT"], prices, lambdas=lambdas)

        assert skipped == ["SHORT"]
        assert list(per_asset.columns) == ["ticker", "lambda", "r2"]
        assert set(per_asset["ticker"]) == {"A", "B"}
        assert list(agg.columns) == ["lambda", "mean_r2", "std_r2", "n_assets"]
        assert len(agg) == len(lambdas)
        assert (agg["n_assets"] == 2).all()
        for _, row in agg.iterrows():
            assert row["mean_r2"] == pytest.approx(1.0 / row["lambda"])
            assert row["std_r2"] == pytest.approx(0.0, abs=1e-9)

    def test_skips_tickers_with_insufficient_history(self):
        n_rows = RIDGE_LOOKBACK_H + RIDGE_FORECAST_HORIZON  # shorter than RIDGE_MIN_ROWS
        dates = pd.bdate_range("2000-01-01", periods=n_rows)
        prices = pd.DataFrame({"SHORT": np.linspace(10, 11, n_rows)}, index=dates)

        per_asset, agg, skipped = ridge_r2_sweep(["SHORT"], prices, lambdas=[1.0])

        assert skipped == ["SHORT"]
        assert per_asset.empty
        assert agg.empty


class TestRestrictPanel:
    def test_restricts_columns_and_recomputes_listing_index(self):
        class FakePanel:
            pass

        returns = _synthetic_returns_frame(n_days=50, n_assets=4, seed=2)
        returns.iloc[:10, returns.columns.get_loc("T2")] = np.nan  # T2 listed later
        prices = (1.0 + returns.fillna(0.0)).cumprod()
        panel = FakePanel()
        panel.returns = returns
        panel.prices = prices

        sub_returns, sub_prices, listing_index = restrict_panel(panel, ["T2", "T0", "NOT_IN_PANEL"])

        assert list(sub_returns.columns) == ["T2", "T0"]
        assert list(sub_prices.columns) == ["T2", "T0"]
        assert listing_index == listing_day_index(sub_returns)
        assert listing_index["T2"] == 10
        assert listing_index["T0"] == 0
