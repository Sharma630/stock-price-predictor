"""
Handles downloading historical OHLCV data from Yahoo Finance (via yfinance),
with a bundled-sample-data fallback so the app can still be demonstrated
when live downloading fails (no internet, Yahoo temporarily down, etc.).

This module never silently fabricates market prices. If live data cannot be
retrieved, it falls back to the bundled sample CSV and the caller is expected
to surface `used_sample_data=True` to the UI so the user is informed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Tuple

import pandas as pd

from config import settings
from src.utils import DataUnavailableError, InvalidConfigurationError, get_logger

logger = get_logger(__name__)


@dataclass
class LoadResult:
    """Container describing where the returned data came from."""

    data: pd.DataFrame
    used_sample_data: bool
    source_message: str


def validate_date_range(start_date: date, end_date: date) -> None:
    """Raise InvalidConfigurationError for obviously bad date ranges."""
    if start_date >= end_date:
        raise InvalidConfigurationError("Start date must be earlier than end date.")
    if end_date > date.today():
        raise InvalidConfigurationError("End date cannot be in the future.")
    if (end_date - start_date).days < 30:
        raise InvalidConfigurationError(
            "Date range is too short. Please select at least a few weeks of data."
        )


def _download_from_yahoo(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance. Raises DataUnavailableError on failure."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment issue
        raise DataUnavailableError(
            "The 'yfinance' package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        df = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:  # network errors, rate limiting, etc.
        raise DataUnavailableError(
            f"Could not reach Yahoo Finance for {ticker}. "
            f"Check your internet connection and try again. ({exc})"
        ) from exc

    if df is None or df.empty:
        raise DataUnavailableError(
            f"Yahoo Finance returned no data for {ticker} in the selected date range."
        )

    # yfinance sometimes returns a MultiIndex column structure for single tickers
    # depending on version; flatten it defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    missing_cols = [c for c in settings.REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataUnavailableError(
            f"Downloaded data is missing required columns: {missing_cols}"
        )

    return df


def _load_sample_data(index_key: str) -> pd.DataFrame:
    """Load the bundled sample CSV for the given index."""
    config = settings.INDEX_OPTIONS[index_key]
    sample_path = settings.SAMPLE_DATA_DIR / config["sample_csv"]
    if not sample_path.exists():
        raise DataUnavailableError(
            f"No live data available and no bundled sample dataset found for "
            f"{config['label']} at {sample_path}."
        )
    df = pd.read_csv(sample_path, parse_dates=["Date"])
    return df


def load_market_data(
    index_key: str,
    start_date: date,
    end_date: date,
    allow_sample_fallback: bool = True,
) -> LoadResult:
    """
    Load OHLCV data for the given index between start_date and end_date.

    Tries Yahoo Finance first. If that fails and allow_sample_fallback is
    True, falls back to the bundled sample dataset (clearly flagged).
    Raises DataUnavailableError if neither source works.
    """
    if index_key not in settings.INDEX_OPTIONS:
        raise InvalidConfigurationError(f"Unknown index selection: {index_key}")

    validate_date_range(start_date, end_date)
    ticker = settings.INDEX_OPTIONS[index_key]["yf_ticker"]

    try:
        df = _download_from_yahoo(ticker, start_date, end_date)
        logger.info("Downloaded %d rows for %s from Yahoo Finance.", len(df), ticker)
        return LoadResult(data=df, used_sample_data=False, source_message="Live data from Yahoo Finance.")
    except DataUnavailableError as exc:
        logger.warning("Live download failed for %s: %s", ticker, exc)
        if not allow_sample_fallback:
            raise
        try:
            df = _load_sample_data(index_key)
            # Filter sample data to the requested window when possible.
            mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
            filtered = df.loc[mask].reset_index(drop=True)
            df_final = filtered if len(filtered) >= 30 else df
            return LoadResult(
                data=df_final,
                used_sample_data=True,
                source_message=settings.SAMPLE_DATA_NOTICE,
            )
        except DataUnavailableError:
            raise DataUnavailableError(
                "Live data download failed and no bundled sample dataset is "
                "available for this index. Please check your internet connection."
            ) from exc
