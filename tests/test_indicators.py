"""Unit tests for src/indicators.py — SMA, EMA, MACD, RSI."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import add_all_indicators, ema, macd, rsi, sma


@pytest.fixture
def price_series() -> pd.Series:
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 200))
    return pd.Series(prices)


def test_sma_matches_manual_average(price_series):
    result = sma(price_series, window=5)
    manual = price_series.rolling(5).mean()
    pd.testing.assert_series_equal(result, manual, check_names=False)


def test_sma_has_nan_warmup(price_series):
    result = sma(price_series, window=10)
    assert result.iloc[:9].isna().all()
    assert result.iloc[9:].notna().all()


def test_ema_reacts_faster_than_sma_of_same_span():
    # A step change should move EMA more than SMA of comparable length.
    series = pd.Series([10] * 30 + [20] * 30)
    ema_result = ema(series, span=10)
    sma_result = sma(series, window=10)
    idx = 35  # a few steps after the jump
    assert abs(ema_result.iloc[idx] - 20) < abs(sma_result.iloc[idx] - 20)


def test_macd_columns_present_and_consistent(price_series):
    result = macd(price_series)
    assert list(result.columns) == ["MACD", "MACD_SIGNAL", "MACD_HIST"]
    # Histogram should equal MACD minus Signal wherever both are defined.
    valid = result.dropna()
    np.testing.assert_allclose(
        (valid["MACD"] - valid["MACD_SIGNAL"]).values,
        valid["MACD_HIST"].values,
        atol=1e-8,
    )


def test_rsi_within_valid_range(price_series):
    result = rsi(price_series, period=14).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_rsi_is_high_for_strictly_increasing_series():
    series = pd.Series(np.arange(1, 50, dtype=float))
    result = rsi(series, period=14).dropna()
    # A monotonically increasing series has no losses -> RSI should be 100.
    assert (result > 95).all()


def test_add_all_indicators_adds_expected_columns():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=200, freq="B"),
            "Open": np.linspace(100, 120, 200),
            "High": np.linspace(101, 121, 200),
            "Low": np.linspace(99, 119, 200),
            "Close": np.linspace(100, 120, 200),
            "Volume": np.random.default_rng(1).integers(1000, 5000, 200),
        }
    )
    enriched = add_all_indicators(df)
    for col in ["SMA_20", "SMA_50", "EMA_12", "EMA_26", "MACD", "MACD_SIGNAL", "MACD_HIST", "RSI_14"]:
        assert col in enriched.columns
