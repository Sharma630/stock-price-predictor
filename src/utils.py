"""
Small shared helpers used across the project: custom exceptions, logging
setup, and generic utility functions. Centralizing these avoids duplicated
error-handling logic in the Streamlit pages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# --------------------------------------------------------------------------
# Custom exceptions
# --------------------------------------------------------------------------
class DataUnavailableError(Exception):
    """Raised when market data cannot be downloaded and no fallback exists."""


class InsufficientDataError(Exception):
    """Raised when there are not enough rows to build sequences / train."""


class ModelNotFoundError(Exception):
    """Raised when a prediction is requested but no trained model exists."""


class InvalidConfigurationError(Exception):
    """Raised when user-provided settings (dates, lookback, etc.) are invalid."""


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent, simple format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def safe_read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON metadata file, raising a friendly error if it is missing/corrupt."""
    import json

    if not path.exists():
        raise ModelNotFoundError(f"Metadata file not found: {path.name}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ModelNotFoundError(f"Metadata file is corrupted: {path.name}") from exc


def format_index_points(value: float) -> str:
    """Format a numeric index value for display (e.g. 22,345.67)."""
    return f"{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a fraction (0.0123) as a signed percentage string (+1.23%)."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "dataset") -> None:
    """Raise InsufficientDataError with a clear message if df is empty/None."""
    if df is None or df.empty:
        raise InsufficientDataError(
            f"The {context} is empty. Try a wider date range or check your "
            f"internet connection."
        )
