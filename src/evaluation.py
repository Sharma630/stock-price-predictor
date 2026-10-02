"""
Model evaluation metrics: MAE, RMSE, MAPE, R^2, and Directional Accuracy.

All functions operate on real (inverse-scaled) index-point values, not the
normalized [0, 1] scaled values, so the metrics are directly interpretable.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of consecutive steps where the predicted direction of change
    (up/down relative to the previous actual value) matches the actual
    direction of change.
    """
    if len(y_true) < 2:
        return float("nan")

    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(y_pred[1:] - y_true[:-1])

    matches = actual_direction == predicted_direction
    return float(np.mean(matches) * 100)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE, guarding against division by zero."""
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    errors = np.abs((y_true - y_pred) / y_true_safe)
    return float(np.nanmean(errors) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the full evaluation metric suite required by the project spec."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    dir_acc = directional_accuracy(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(float(r2), 4),
        "Directional_Accuracy": round(dir_acc, 2),
    }


METRIC_EXPLANATIONS = {
    "MAE": "Average absolute difference between actual and predicted closing values.",
    "RMSE": "Root Mean Squared Error — penalizes larger prediction errors more heavily than MAE.",
    "MAPE": "Mean Absolute Percentage Error — average error expressed as a percentage of the actual value.",
    "R2": "R-squared — indicates how much of the variation in the target is captured by the model (closer to 1 is better).",
    "Directional_Accuracy": "Percentage of cases where the predicted UP/DOWN movement matched the actual direction.",
}
