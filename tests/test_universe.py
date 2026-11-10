"""Unit tests for src/universe.py (paper Table 1, Table 2)."""

from __future__ import annotations

import pytest

from src.universe import (
    CATEGORIES,
    FULL_UNIVERSE,
    SEARCH_UNIVERSE_23,
    asset_class_of,
    category_of,
    universe_excluding_categories,
)


def test_full_universe_has_59_unique_etfs():
    assert len(FULL_UNIVERSE) == 59
    assert len(set(FULL_UNIVERSE)) == 59


def test_every_category_maps_to_one_asset_class():
    for ticker in FULL_UNIVERSE:
        cat = category_of(ticker)
        asset_class = asset_class_of(ticker)
        assert asset_class in {"Equities", "Fixed income", "Alternatives"}
        assert ticker in CATEGORIES[cat]


def test_unknown_ticker_raises():
    with pytest.raises(KeyError):
        category_of("NOT_A_REAL_ETF")


def test_universe_excluding_categories_removes_requested_tickers():
    reduced = universe_excluding_categories(("Country", "Futures"))
    assert "EWJ" not in reduced  # Country
    assert "CORN" not in reduced  # Futures
    assert "SPY" in reduced
    assert len(reduced) == len(FULL_UNIVERSE) - len(CATEGORIES["Country"]) - len(CATEGORIES["Futures"])


def test_search_universe_23_excludes_country_and_futures():
    assert set(SEARCH_UNIVERSE_23).isdisjoint(CATEGORIES["Country"])
    assert set(SEARCH_UNIVERSE_23).isdisjoint(CATEGORIES["Futures"])
    assert "SPY" in SEARCH_UNIVERSE_23 and "GLD" in SEARCH_UNIVERSE_23
