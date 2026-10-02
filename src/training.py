"""
Orchestrates the full training pipeline: clean -> features -> scale ->
sequences -> train LSTM -> evaluate -> save model + scaler + metadata.

Training is only ever triggered explicitly (a Train button in the UI, or
a direct call to `train_pipeline`) — never automatically on app startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from config import settings
from src.evaluation import evaluate_predictions
from src.feature_engineering import build_feature_dataframe, fit_scaler_on_training_data, save_scaler
from src.model import build_lstm_model, get_training_callbacks
from src.preprocessing import clean_ohlcv
from src.sequences import create_sequences
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    history: Dict[str, list]
    metrics: Dict[str, float]
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int
    n_features: int
    lookback: int
    feature_columns: list
    trained_at: str
    date_range: Dict[str, str]


def train_pipeline(
    raw_df: pd.DataFrame,
    index_key: str,
    lookback: int = settings.DEFAULT_LOOKBACK,
    epochs: int = settings.DEFAULT_EPOCHS,
    batch_size: int = settings.DEFAULT_BATCH_SIZE,
    lstm_units_1: int = settings.DEFAULT_LSTM_UNITS_1,
    lstm_units_2: int = settings.DEFAULT_LSTM_UNITS_2,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> "TrainingResult":
    """
    Run the end-to-end training pipeline and persist the trained model,
    scaler, and metadata to disk using ticker-specific filenames.
    """

    def report(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    report("Cleaning raw market data...")
    clean_df = clean_ohlcv(raw_df)

    report("Calculating technical indicators (SMA, EMA, MACD, RSI)...")
    feature_df = build_feature_dataframe(clean_df)

    report("Splitting data chronologically and fitting scaler on training data only...")
    scaler, split = fit_scaler_on_training_data(feature_df)

    target_col_index = settings.FEATURE_COLUMNS.index(settings.TARGET_COLUMN)

    train_scaled = scaler.transform(split.train[settings.FEATURE_COLUMNS].values)
    val_scaled = scaler.transform(split.val[settings.FEATURE_COLUMNS].values)
    test_scaled = scaler.transform(split.test[settings.FEATURE_COLUMNS].values)

    report(f"Building sliding-window sequences (lookback={lookback})...")
    X_train, y_train = create_sequences(train_scaled, target_col_index, lookback)
    X_val, y_val = create_sequences(val_scaled, target_col_index, lookback)
    X_test, y_test = create_sequences(test_scaled, target_col_index, lookback)

    n_features = X_train.shape[2]

    report("Building LSTM model...")
    model = build_lstm_model(
        lookback=lookback,
        n_features=n_features,
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
    )

    report(f"Training for up to {epochs} epochs (batch_size={batch_size})...")
    callbacks = get_training_callbacks()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    report("Evaluating on the held-out test set...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform predictions and actuals back to real index-point scale.
    y_test_real = _inverse_transform_target(scaler, test_scaled, y_test, target_col_index)
    y_pred_real = _inverse_transform_target(scaler, test_scaled, y_pred_scaled, target_col_index)

    metrics = evaluate_predictions(y_test_real, y_pred_real)

    report("Saving model, scaler, and metadata...")
    paths = settings.model_paths(index_key)
    model.save(paths["model"])
    save_scaler(scaler, paths["scaler"])

    metadata = {
        "index_key": index_key,
        "ticker": settings.INDEX_OPTIONS[index_key]["yf_ticker"],
        "feature_columns": settings.FEATURE_COLUMNS,
        "target_column": settings.TARGET_COLUMN,
        "lookback": lookback,
        "epochs_requested": epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "batch_size": batch_size,
        "lstm_units_1": lstm_units_1,
        "lstm_units_2": lstm_units_2,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "date_range": {
            "start": str(clean_df["Date"].min().date()),
            "end": str(clean_df["Date"].max().date()),
        },
        "n_train_samples": int(len(X_train)),
        "n_val_samples": int(len(X_val)),
        "n_test_samples": int(len(X_test)),
        "metrics": metrics,
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report("Training complete.")

    return TrainingResult(
        history=history.history,
        metrics=metrics,
        n_train_samples=len(X_train),
        n_val_samples=len(X_val),
        n_test_samples=len(X_test),
        n_features=n_features,
        lookback=lookback,
        feature_columns=settings.FEATURE_COLUMNS,
        trained_at=metadata["trained_at"],
        date_range=metadata["date_range"],
    )


def _inverse_transform_target(scaler, scaled_reference_rows, target_scaled_values, target_col_index):
    """
    MinMaxScaler was fit on all feature columns jointly, so to inverse-transform
    just the target column we reconstruct a dummy row per value, placing the
    target value in the correct column position, then invert and slice it out.
    """
    n_features = scaled_reference_rows.shape[1]
    dummy = np.zeros((len(target_scaled_values), n_features))
    dummy[:, target_col_index] = target_scaled_values
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, target_col_index]
