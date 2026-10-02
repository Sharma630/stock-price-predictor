"""
Builds sliding-window sequences for the LSTM model.

Given a scaled feature matrix of shape (n_rows, n_features), we build
overlapping windows of length `lookback`, where the target for each window
is the (scaled) Close value of the row immediately AFTER the window ends.

To avoid leakage across the train/validation/test boundary, sequences must
be built from CONTIGUOUS scaled arrays that already respect the chronological
split — i.e. call this once per split, not on the concatenated full dataset.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config import settings
from src.utils import InsufficientDataError


def create_sequences(
    scaled_features: np.ndarray,
    target_col_index: int,
    lookback: int = settings.DEFAULT_LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D array (rows=time, cols=features) into 3D LSTM input sequences.

    Parameters
    ----------
    scaled_features : np.ndarray of shape (n_rows, n_features)
    target_col_index : index of the column to use as the prediction target
        (e.g. the scaled 'Close' column) within `scaled_features`.
    lookback : number of past time steps in each input sequence.

    Returns
    -------
    X : np.ndarray of shape (n_samples, lookback, n_features)
    y : np.ndarray of shape (n_samples,) — the scaled target for the step
        immediately following each window.
    """
    n_rows = scaled_features.shape[0]

    # We need at least `lookback` rows to form one window, plus one more row
    # to serve as that window's target.
    if n_rows <= lookback:
        raise InsufficientDataError(
            f"Not enough rows ({n_rows}) to build sequences with lookback={lookback}. "
            f"Reduce the lookback or select a wider date range."
        )

    X, y = [], []
    for i in range(lookback, n_rows):
        X.append(scaled_features[i - lookback : i, :])
        y.append(scaled_features[i, target_col_index])

    return np.array(X), np.array(y)


def last_window(
    scaled_features: np.ndarray,
    lookback: int = settings.DEFAULT_LOOKBACK,
) -> np.ndarray:
    """
    Return the most recent `lookback` rows as a single sequence, shaped
    (1, lookback, n_features), ready to feed into model.predict() for a
    next-step forecast.
    """
    if scaled_features.shape[0] < lookback:
        raise InsufficientDataError(
            f"Need at least {lookback} rows of recent data to forecast; "
            f"only {scaled_features.shape[0]} are available."
        )
    window = scaled_features[-lookback:, :]
    return window.reshape(1, lookback, scaled_features.shape[1])
