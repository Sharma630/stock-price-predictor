"""Unit tests for src/sequences.py — LSTM sliding-window sequence generation."""

import numpy as np
import pytest

from src.sequences import create_sequences, last_window
from src.utils import InsufficientDataError


def test_create_sequences_output_shape():
    n_rows, n_features, lookback = 200, 5, 20
    data = np.random.default_rng(0).random((n_rows, n_features))
    X, y = create_sequences(data, target_col_index=3, lookback=lookback)

    expected_samples = n_rows - lookback
    assert X.shape == (expected_samples, lookback, n_features)
    assert y.shape == (expected_samples,)


def test_create_sequences_target_alignment():
    # Use a feature matrix where column 0 is just the row index, so we can
    # verify each window's target is exactly the value right after it.
    n_rows, lookback = 50, 10
    data = np.tile(np.arange(n_rows).reshape(-1, 1), (1, 3)).astype(float)
    X, y = create_sequences(data, target_col_index=0, lookback=lookback)

    # First window covers rows [0..9], target should be row 10's value (10).
    assert X[0, -1, 0] == 9
    assert y[0] == 10
    # Last window's target should be the final row's value.
    assert y[-1] == n_rows - 1


def test_create_sequences_raises_when_not_enough_rows():
    data = np.random.default_rng(0).random((10, 3))
    with pytest.raises(InsufficientDataError):
        create_sequences(data, target_col_index=0, lookback=20)


def test_last_window_shape_and_content():
    n_rows, n_features, lookback = 100, 4, 15
    data = np.random.default_rng(0).random((n_rows, n_features))
    window = last_window(data, lookback=lookback)

    assert window.shape == (1, lookback, n_features)
    np.testing.assert_allclose(window[0], data[-lookback:, :])


def test_last_window_raises_when_not_enough_rows():
    data = np.random.default_rng(0).random((5, 3))
    with pytest.raises(InsufficientDataError):
        last_window(data, lookback=10)
