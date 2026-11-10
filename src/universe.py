"""ETF universe and category definitions (paper Table 1 and Table 2, Sec. 2.1-2.2).

The 59-ETF universe spans three asset classes split into nine structural
categories. Categories are used both to build filtered search universes
(Sec. 6.2) and to run the category-ablation experiment (Sec. 6.3).
"""

from __future__ import annotations

# Table 2: nine categories grouping the 59 ETFs of Table 1 by asset class and
# underlying structure.
CATEGORIES: dict[str, list[str]] = {
    "Market": ["SPY", "QQQ", "VTV", "VUG", "MDY", "IWM", "SCHD", "USMV", "QUAL"],
    "Sector": ["XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB", "IBB", "IYR"],
    "Country": [
        "EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY",
        "EWP", "EWW", "EWI", "EWD", "EWL", "EWC",
    ],
    "Government": ["AGG", "TLT", "IEF", "TIP", "MUB"],
    "Corporate": ["LQD", "LQDH", "HYG"],
    "Global": ["BNDX", "EMB", "IAGG", "VWOB"],
    "Physical": ["GLD", "SLV", "PPLT"],
    "Futures": ["CPER", "USO", "UGA", "CORN", "WEAT", "SOYB", "CANE"],
    "Crypto": ["IBIT", "ETHA"],
}

ASSET_CLASS_OF_CATEGORY: dict[str, str] = {
    "Market": "Equities", "Sector": "Equities", "Country": "Equities",
    "Government": "Fixed income", "Corporate": "Fixed income", "Global": "Fixed income",
    "Physical": "Alternatives", "Futures": "Alternatives", "Crypto": "Alternatives",
}

FULL_UNIVERSE: list[str] = [t for tickers in CATEGORIES.values() for t in tickers]
assert len(FULL_UNIVERSE) == 59, f"expected 59 ETFs, got {len(FULL_UNIVERSE)}"


def category_of(ticker: str) -> str:
    """Return the Table 2 category of ``ticker``."""
    for category, tickers in CATEGORIES.items():
        if ticker in tickers:
            return category
    raise KeyError(f"{ticker} is not part of the 59-ETF universe (Table 1)")


def asset_class_of(ticker: str) -> str:
    """Return the asset class (Equities / Fixed income / Alternatives) of ``ticker``."""
    return ASSET_CLASS_OF_CATEGORY[category_of(ticker)]


def universe_excluding_categories(categories_to_exclude: tuple[str, ...]) -> list[str]:
    """Return the universe with the given Table 2 categories removed."""
    return [t for t, cat in ((t, category_of(t)) for t in FULL_UNIVERSE) if cat not in categories_to_exclude]


# Sec. 6.2: the 23-ETF filtered search universe excludes the Country and Futures
# categories (config.EXCLUDED_CATEGORIES_FOR_SEARCH) for the exhaustive equal-weight
# and inverse-volatility searches. The remaining ETFs are further restricted, at
# runtime, to those already listed at the start of the backtest window (Sec. 6.2),
# which is checked against each ETF's first available trading date rather than
# hand-picked here.
SEARCH_UNIVERSE_23 = universe_excluding_categories(("Country", "Futures"))

CASH_RATE_SERIES_NAME = "effective_fed_funds_rate"  # Massive /fed/v1/funding-conditions field used as r_f (Table 3)
