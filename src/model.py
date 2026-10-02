"""
Defines the LSTM deep-learning architecture used to forecast the next
closing value from a multivariate sequence of technical-indicator features.

Architecture (see docs/architecture.md for the full diagram):

    Input(lookback, n_features)
      -> LSTM(units_1, return_sequences=True)
      -> Dropout(dropout)
      -> LSTM(units_2)
      -> Dropout(dropout)
      -> Dense(dense_units, activation="relu")
      -> Dense(1)                                   [linear output]

TensorFlow/Keras is imported lazily inside functions so the rest of the
codebase (indicators, preprocessing, sequence generation) can be imported
and unit-tested even in environments where TensorFlow is not yet installed.
"""

from __future__ import annotations

from typing import Any

from config import settings


def build_lstm_model(
    lookback: int,
    n_features: int,
    lstm_units_1: int = settings.DEFAULT_LSTM_UNITS_1,
    lstm_units_2: int = settings.DEFAULT_LSTM_UNITS_2,
    dropout: float = settings.DEFAULT_DROPOUT,
    dense_units: int = settings.DEFAULT_DENSE_UNITS,
    learning_rate: float = settings.DEFAULT_LEARNING_RATE,
) -> Any:
    """Build and compile the LSTM regression model. Returns a tf.keras.Model."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(lstm_units_1, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units_2),
            layers.Dropout(dropout),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(1),
        ],
        name="stock_index_lstm",
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def get_training_callbacks(patience: int = 5) -> list:
    """Standard EarlyStopping + ReduceLROnPlateau callbacks for lightweight CPU training."""
    from tensorflow import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
        ),
    ]


def load_keras_model(path) -> Any:
    """Load a saved .keras model from disk."""
    from tensorflow import keras

    return keras.models.load_model(path)
