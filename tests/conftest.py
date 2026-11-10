"""Shared pytest fixtures and marker registration.

Market-data tests load a reduced slice of the paper's universe through
``data/data_loader.py`` and cache it with pytest's own cache directory so
repeated test runs do not re-download it.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.register_assert_rewrite("data.data_loader")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: requires DATALOADER_API_KEY/MASSIVE_API_KEY and a live network call"
    )


@pytest.fixture(scope="session")
def has_api_keys() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def market_slice(request: pytest.FixtureRequest, has_api_keys: bool):
    """A small (3-ETF, 2 year) slice of real market data, cached across runs."""
    if not has_api_keys:
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")

    from data.data_loader import load_adjusted_close_panel, load_fed_funds_rate

    cache_dir = request.config.cache.mkdir("market_slice")
    prices_path = cache_dir / "prices.csv"
    ffr_path = cache_dir / "ffr.csv"

    if prices_path.exists() and ffr_path.exists():
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        ffr = pd.read_csv(ffr_path, index_col=0, parse_dates=True)["rate"]
        return prices, ffr

    tickers = ["SPY", "AGG", "GLD"]
    start, end = "2018-01-01", "2019-12-31"
    prices, _meta = load_adjusted_close_panel(tickers, start, end, identity_date=end)
    ffr = load_fed_funds_rate(start, end)
    prices.to_csv(prices_path)
    ffr.rename("rate").to_csv(ffr_path)
    return prices, ffr


@pytest.fixture
def synthetic_returns() -> np.ndarray:
    """500 days x 6 assets of iid Gaussian daily returns, fixed seed."""
    rng = np.random.default_rng(42)
    n_days, n_assets = 500, 6
    vols = np.array([0.01, 0.012, 0.008, 0.02, 0.015, 0.009])
    corr = np.full((n_assets, n_assets), 0.2)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, n_assets))
    return (z @ chol.T) * vols
