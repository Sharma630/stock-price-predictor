"""Unit tests for src/preprocessing.py — cleaning and chronological splitting."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import chronological_split, clean_ohlcv, drop_indicator_warmup_rows
from src.utils import InsufficientDataError


def _make_raw_df(n=300, shuffle=False, add_dupes=False, add_nans=False):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + abs(rng.normal(1, 0.5, n)),
            "Low": close - abs(rng.normal(1, 0.5, n)),
            "Close": close,
            "Volume": rng.integers(1000, 5000, n).astype(float),
        }
    )
    if add_dupes:
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    if add_nans:
        df.loc[10, "Close"] = np.nan
    if shuffle:
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    return df


def test_clean_ohlcv_sorts_chronologically():
    df = _make_raw_df(shuffle=True)
    cleaned = clean_ohlcv(df)
    assert cleaned["Date"].is_monotonic_increasing


def test_clean_ohlcv_removes_duplicate_dates():
    df = _make_raw_df(add_dupes=True)
    cleaned = clean_ohlcv(df)
    assert cleaned["Date"].duplicated().sum() == 0


def test_clean_ohlcv_handles_missing_values():
    df = _make_raw_df(add_nans=True)
    cleaned = clean_ohlcv(df)
    assert cleaned["Close"].isna().sum() == 0


def test_clean_ohlcv_raises_on_too_few_rows():
    df = _make_raw_df(n=50)
    with pytest.raises(InsufficientDataError):
        clean_ohlcv(df)


def test_clean_ohlcv_fills_missing_volume_column():
    df = _make_raw_df(n=300).drop(columns=["Volume"])
    cleaned = clean_ohlcv(df)
    assert "Volume" in cleaned.columns
    assert (cleaned["Volume"] == 0.0).all()


def test_chronological_split_preserves_order_and_ratios():
    df = _make_raw_df(n=1000)
    split = chronological_split(df, train_ratio=0.7, val_ratio=0.15)

    assert len(split.train) + len(split.val) + len(split.test) == len(df)
    assert split.train["Date"].max() < split.val["Date"].min()
    assert split.val["Date"].max() < split.test["Date"].min()
    assert abs(len(split.train) / len(df) - 0.7) < 0.01


def test_chronological_split_raises_on_empty_frame():
    df = pd.DataFrame(columns=["Date", "Close"])
    with pytest.raises(InsufficientDataError):
        chronological_split(df)


def test_drop_indicator_warmup_rows():
    df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [np.nan, 2, 3, 4]})
    cleaned = drop_indicator_warmup_rows(df, ["A", "B"])
    assert len(cleaned) == 2
    assert cleaned["A"].isna().sum() == 0
