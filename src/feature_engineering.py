"""
Feature engineering: turns cleaned OHLCV + indicator data into the final
feature matrix used for scaling and sequence generation, and fits/saves
the MinMaxScaler on the TRAINING split only (to avoid data leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import settings
from src.indicators import add_all_indicators
from src.preprocessing import ChronologicalSplit, chronological_split, drop_indicator_warmup_rows
from src.utils import InsufficientDataError, get_logger

logger = get_logger(__name__)


def build_feature_dataframe(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators, then drop warm-up rows so every remaining row has full features."""
    enriched = add_all_indicators(clean_df)
    enriched = drop_indicator_warmup_rows(enriched, settings.FEATURE_COLUMNS)
    if len(enriched) < settings.MIN_REQUIRED_ROWS // 2:
        raise InsufficientDataError(
            "Not enough rows remain after computing technical indicators "
            "(they need a warm-up period). Please select a wider date range."
        )
    return enriched


def fit_scaler_on_training_data(
    feature_df: pd.DataFrame,
) -> Tuple[MinMaxScaler, ChronologicalSplit]:
    """
    Chronologically split the feature dataframe, then fit a MinMaxScaler
    using ONLY the training portion (never validation/test) to prevent
    data leakage from the future into the past.
    """
    split = chronological_split(feature_df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(split.train[settings.FEATURE_COLUMNS].values)

    return scaler, split


def transform_split(scaler: MinMaxScaler, df: pd.DataFrame) -> np.ndarray:
    """Apply an already-fitted scaler to a dataframe's feature columns."""
    return scaler.transform(df[settings.FEATURE_COLUMNS].values)


def save_scaler(scaler: MinMaxScaler, path: Path) -> None:
    joblib.dump(scaler, path)
    logger.info("Saved scaler to %s", path)


def load_scaler(path: Path) -> MinMaxScaler:
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)
