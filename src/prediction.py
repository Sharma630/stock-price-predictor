"""
Loads a trained model bundle (model + scaler + metadata) and generates
next-step and experimental multi-step forecasts, including UP/DOWN/NEUTRAL
direction classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config import settings
from src.evaluation import evaluate_predictions
from src.feature_engineering import build_feature_dataframe, load_scaler
from src.model import load_keras_model
from src.preprocessing import chronological_split, clean_ohlcv
from src.sequences import create_sequences, last_window
from src.utils import ModelNotFoundError, get_logger, safe_read_json

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    model: object
    scaler: object
    metadata: dict


def model_exists(index_key: str) -> bool:
    paths = settings.model_paths(index_key)
    return paths["model"].exists() and paths["scaler"].exists() and paths["metadata"].exists()


def load_model_bundle(index_key: str) -> ModelBundle:
    """Load the saved model, scaler, and metadata for an index. Raises ModelNotFoundError if missing."""
    paths = settings.model_paths(index_key)
    if not model_exists(index_key):
        raise ModelNotFoundError(
            f"No trained model found for {settings.INDEX_OPTIONS[index_key]['label']}. "
            f"Please train a model first on the Model Training page."
        )
    metadata = safe_read_json(paths["metadata"])
    scaler = load_scaler(paths["scaler"])
    model = load_keras_model(paths["model"])
    return ModelBundle(model=model, scaler=scaler, metadata=metadata)


def classify_direction(latest_close: float, predicted_close: float) -> str:
    """UP / DOWN / NEUTRAL classification using a small threshold to avoid noise."""
    change_fraction = (predicted_close - latest_close) / latest_close
    if abs(change_fraction) < settings.DIRECTION_NEUTRAL_THRESHOLD:
        return "NEUTRAL"
    return "UP" if change_fraction > 0 else "DOWN"


def _inverse_transform_single(scaler, scaled_value: float, feature_columns: List[str], target_col: str) -> float:
    idx = feature_columns.index(target_col)
    dummy = np.zeros((1, len(feature_columns)))
    dummy[0, idx] = scaled_value
    return float(scaler.inverse_transform(dummy)[0, idx])


def generate_forecast(raw_df: pd.DataFrame, index_key: str) -> Dict:
    """
    Generate a single next-step forecast using the latest available data
    and the previously trained model bundle for this index.
    """
    bundle = load_model_bundle(index_key)
    feature_columns = bundle.metadata["feature_columns"]
    lookback = bundle.metadata["lookback"]

    clean_df = clean_ohlcv(raw_df)
    feature_df = build_feature_dataframe(clean_df)

    scaled = bundle.scaler.transform(feature_df[feature_columns].values)
    window = last_window(scaled, lookback=lookback)

    pred_scaled = bundle.model.predict(window, verbose=0).flatten()[0]
    predicted_close = _inverse_transform_single(
        bundle.scaler, pred_scaled, feature_columns, settings.TARGET_COLUMN
    )

    latest_row = feature_df.iloc[-1]
    latest_close = float(latest_row["Close"])
    latest_date = latest_row["Date"]

    change = predicted_close - latest_close
    change_pct = change / latest_close
    direction = classify_direction(latest_close, predicted_close)

    return {
        "index_key": index_key,
        "index_label": settings.INDEX_OPTIONS[index_key]["label"],
        "latest_date": str(pd.Timestamp(latest_date).date()),
        "latest_close": latest_close,
        "predicted_close": predicted_close,
        "change": change,
        "change_pct": change_pct,
        "direction": direction,
        "lookback": lookback,
        "model_trained_at": bundle.metadata.get("trained_at"),
        "currency_label": settings.INDEX_OPTIONS[index_key]["currency_label"],
    }


def generate_multi_day_forecast(raw_df: pd.DataFrame, index_key: str, horizons: List[int] = None) -> List[Dict]:
    """
    Experimental recursive multi-step forecast: feed each prediction back in
    as the newest 'Close' value (holding other features at their last known
    values) to produce forecasts for several days ahead. Uncertainty compounds
    with each additional step — this is clearly labeled as experimental in the UI.
    """
    horizons = horizons or settings.MULTI_DAY_HORIZONS
    max_horizon = max(horizons)

    bundle = load_model_bundle(index_key)
    feature_columns = bundle.metadata["feature_columns"]
    lookback = bundle.metadata["lookback"]
    target_idx = feature_columns.index(settings.TARGET_COLUMN)

    clean_df = clean_ohlcv(raw_df)
    feature_df = build_feature_dataframe(clean_df)
    scaled = bundle.scaler.transform(feature_df[feature_columns].values)

    working = scaled.copy()
    results = []
    latest_close = float(feature_df.iloc[-1]["Close"])

    for step in range(1, max_horizon + 1):
        window = last_window(working, lookback=lookback)
        pred_scaled = bundle.model.predict(window, verbose=0).flatten()[0]

        # Build the next synthetic row: carry forward the last row's other
        # features, but override the target (Close) column with the new prediction.
        next_row = working[-1].copy()
        next_row[target_idx] = pred_scaled
        working = np.vstack([working, next_row])

        if step in horizons:
            predicted_close = _inverse_transform_single(
                bundle.scaler, pred_scaled, feature_columns, settings.TARGET_COLUMN
            )
            results.append(
                {
                    "horizon_days": step,
                    "predicted_close": predicted_close,
                    "change_pct": (predicted_close - latest_close) / latest_close,
                    "direction": classify_direction(latest_close, predicted_close),
                }
            )

    return results


def evaluate_saved_model_on_test_set(raw_df: pd.DataFrame, index_key: str) -> Dict:
    """
    Reconstruct the chronological test split for the given index using the
    same feature pipeline as training, run the saved model over it, and
    return dates/actual/predicted arrays plus recomputed metrics — used by
    the Model Performance page (no retraining involved).
    """
    bundle = load_model_bundle(index_key)
    feature_columns = bundle.metadata["feature_columns"]
    lookback = bundle.metadata["lookback"]
    target_idx = feature_columns.index(settings.TARGET_COLUMN)

    clean_df = clean_ohlcv(raw_df)
    feature_df = build_feature_dataframe(clean_df)
    split = chronological_split(feature_df)

    test_scaled = bundle.scaler.transform(split.test[feature_columns].values)
    X_test, y_test_scaled = create_sequences(test_scaled, target_idx, lookback)

    if len(X_test) == 0:
        raise ModelNotFoundError(
            "Not enough test-set rows in the current date range to evaluate the "
            "saved model. Try widening the date range."
        )

    y_pred_scaled = bundle.model.predict(X_test, verbose=0).flatten()

    dummy_true = np.zeros((len(y_test_scaled), len(feature_columns)))
    dummy_true[:, target_idx] = y_test_scaled
    y_true_real = bundle.scaler.inverse_transform(dummy_true)[:, target_idx]

    dummy_pred = np.zeros((len(y_pred_scaled), len(feature_columns)))
    dummy_pred[:, target_idx] = y_pred_scaled
    y_pred_real = bundle.scaler.inverse_transform(dummy_pred)[:, target_idx]

    test_dates = split.test["Date"].iloc[lookback:].reset_index(drop=True)

    metrics = evaluate_predictions(y_true_real, y_pred_real)

    return {
        "dates": test_dates,
        "y_true": y_true_real,
        "y_pred": y_pred_real,
        "metrics": metrics,
        "metadata": bundle.metadata,
    }
