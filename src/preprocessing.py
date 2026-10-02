"""
Preprocessing pipeline for raw OHLCV data: sorting, de-duplication, missing
value handling, and chronological train/validation/test splitting.

IMPORTANT: time-series data must never be randomly shuffled. All splitting
in this module is strictly chronological to avoid look-ahead / data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from config import settings
from src.utils import InsufficientDataError, get_logger, validate_dataframe_not_empty

logger = get_logger(__name__)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw OHLCV dataframe:
      1. Sort chronologically by Date.
      2. Drop duplicate dates.
      3. Coerce OHLCV columns to numeric.
      4. Forward-fill small gaps.
      5. Drop any remaining unusable rows.
    """
    validate_dataframe_not_empty(df, context="downloaded market data")

    data = df.copy()

    if "Date" not in data.columns:
        raise InsufficientDataError("Expected a 'Date' column in the market data.")

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    data = data.reset_index(drop=True)

    for col in settings.REQUIRED_OHLCV_COLUMNS:
        if col not in data.columns:
            if col == "Volume":
                # Some indices (e.g. SENSEX via certain sources) may not report
                # reliable volume. Handle gracefully instead of failing.
                logger.warning("Volume column missing; filling with 0.")
                data["Volume"] = 0.0
            else:
                raise InsufficientDataError(f"Required column '{col}' is missing from the data.")
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Forward-fill small gaps (e.g. holidays interpolated oddly), then drop
    # any rows that are still unusable (e.g. leading NaNs with nothing to fill from).
    data[settings.REQUIRED_OHLCV_COLUMNS] = data[settings.REQUIRED_OHLCV_COLUMNS].ffill()
    data = data.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    data["Volume"] = data["Volume"].fillna(0.0)

    if len(data) < settings.MIN_REQUIRED_ROWS:
        raise InsufficientDataError(
            f"Only {len(data)} usable rows remain after cleaning; at least "
            f"{settings.MIN_REQUIRED_ROWS} are required for a meaningful model. "
            f"Please select a wider date range."
        )

    return data


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = settings.TRAIN_RATIO,
    val_ratio: float = settings.VAL_RATIO,
) -> ChronologicalSplit:
    """
    Split a chronologically-sorted dataframe into train/validation/test sets
    WITHOUT shuffling, to prevent data leakage from the future into the past.
    """
    n = len(df)
    if n == 0:
        raise InsufficientDataError("Cannot split an empty dataset.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise InsufficientDataError(
            "Not enough data to create non-empty train/validation/test splits. "
            "Please select a wider date range."
        )

    return ChronologicalSplit(train=train_df, val=val_df, test=test_df)


def drop_indicator_warmup_rows(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Drop leading rows that contain NaN because indicators need a warm-up window."""
    cleaned = df.dropna(subset=feature_columns).reset_index(drop=True)
    return cleaned
