"""Market data for the ETF-portfolio replication.

OHLCV bars for every ETF and for the bitcoin backfill series come only from
the QuantX Dataloader REST API (``POST /fetch-ohlcvs``). The daily effective
federal funds rate, used as the cash return r_f,t (Sec. 6.1), is not served by
the Dataloader and comes from the Massive Funding Conditions endpoint
(``GET /fed/v1/funding-conditions``), which republishes the FRED DFF series.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_ROOT = "https://api.massive.com"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {dataloader_key}", "X-Massive-Api-Key": massive_key}


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(method, API_ROOT + route, json=body, headers=headers, timeout=900)
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2 ** attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict], dict]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    notes = {
        "result_id": answer["result"].get("result_id") or answer.get("result_id"),
        "meta": answer.get("meta"),
    }
    return bars, details, notes


def load_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    *,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
    yahoo_backstop: bool = True,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Bars for ``tickers`` over the full inclusive window, and load metadata.

    Returns rows of timestamp (UTC), date, ticker, open, high, low, close,
    volume, sorted by ticker and time, plus metadata with every ticker's
    ``ticker_details`` entry. Tickers that failed are listed under ``failed``;
    the load fails only when no ticker loads.
    """
    request = {
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(
            lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
            enumerate(batches),
        ))
    frames = [bars for bars, _, _ in results if not bars.empty]
    details = [detail for _, batch_details, _ in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request,
        "ticker_details": details,
        "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {
            d["ticker"]: d["data_source"] for d in details
            if d.get("data_source") not in (None, "massive")
        },
        "batches": [notes for _, _, notes in results],
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def load_adjusted_close_panel(
    tickers: list[str], start_date: str, end_date: str, **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Wide (date x ticker) panel of dividend-adjusted close prices."""
    bars, meta = load_ohlcv(tickers, start_date, end_date, **kwargs)
    panel = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    return panel, meta


def load_fed_funds_rate(start_date: str, end_date: str, use_cache: bool = True) -> pd.Series:
    """Daily effective federal funds rate (annualized percentage) from Massive.

    Massive Funding Conditions republishes the FRED DFF series
    (``effective_fed_funds_rate``), the same observation the paper's cash
    leg draws from FRED; it is read directly from Massive rather than
    fredapi because the Dataloader does not serve non-OHLCV series and
    Massive is the designated backstop for such data.
    """
    key = hashlib.sha256(f"ffr-{start_date}-{end_date}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"ffr_{key}.csv"
    if use_cache and cache_path.exists():
        s = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")["rate"]
        return s

    load_dotenv()
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise DataloaderError("MASSIVE_API_KEY must be set")
    rows: list[dict] = []
    url = f"{MASSIVE_ROOT}/fed/v1/funding-conditions"
    params = {
        "date.gte": start_date, "date.lte": end_date, "limit": 50000,
        "sort": "date.asc", "fields": "date,effective_fed_funds_rate",
        "apiKey": api_key,
    }
    while url:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            raise DataloaderError(f"funding-conditions: HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        rows.extend(payload.get("results", []))
        url = payload.get("next_url")
        params = {"apiKey": api_key} if url else None
    if not rows:
        raise DataloaderError("funding-conditions returned no rows")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["effective_fed_funds_rate"].sort_index() / 100.0
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("rate").to_frame().reset_index().to_csv(cache_path, index=False)
    return series


def daily_cash_return(fed_funds_annual: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Convert the annualized FFR (as a decimal) to a daily simple return (Table 3, r_f,t).

    r_f,t = (1 + FFR_t)^(1/252) - 1, reindexed and forward-filled onto the
    ETF trading calendar (the FFR series is posted every calendar day).
    """
    ffr = fed_funds_annual.reindex(trading_dates.union(fed_funds_annual.index)).sort_index().ffill()
    ffr = ffr.reindex(trading_dates)
    return (1.0 + ffr) ** (1.0 / 252.0) - 1.0


def load_bitcoin_backfill(start_date: str, end_date: str, **kwargs) -> pd.Series:
    """Bitcoin spot daily close series used to backfill IBIT before its 2024 listing.

    Loaded through the Dataloader with the crypto ticker ``X:BTCUSD``, which
    bypasses the equity ticker-identity and Yahoo-backstop handling (symbols
    with a market prefix are loaded exactly as given).
    """
    from src.config import IBIT_BACKFILL_SOURCE_TICKER

    panel, _ = load_adjusted_close_panel(
        [IBIT_BACKFILL_SOURCE_TICKER], start_date, end_date, yahoo_backstop=False, **kwargs,
    )
    return panel[IBIT_BACKFILL_SOURCE_TICKER]
