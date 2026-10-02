"""
Technical indicator calculations, implemented directly with pandas/numpy so
the formulas are transparent and easy to explain in a viva.

All functions take/return pandas Series or DataFrames and are pure
(no side effects), which keeps them easy to unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average: the average closing price over the last `window` periods."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average: weights recent prices more heavily than older ones."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(
    series: pd.Series,
    fast: int = settings.MACD_FAST,
    slow: int = settings.MACD_SLOW,
    signal: int = settings.MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.

    MACD line     = EMA(fast) - EMA(slow)
    Signal line   = EMA(MACD line, span=signal)
    Histogram     = MACD line - Signal line
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_SIGNAL": signal_line,
            "MACD_HIST": histogram,
        }
    )


def rsi(series: pd.Series, period: int = settings.RSI_PERIOD) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing method.

    RSI = 100 - (100 / (1 + RS)), where RS = average gain / average loss
    over the given period. Ranges approximately between 0 and 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing is equivalent to an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (no losses in window), RSI is defined as 100.
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values


def rsi_interpretation(value: float) -> str:
    """Return a plain-language interpretation for a single RSI value."""
    if pd.isna(value):
        return "Not enough data"
    if value > settings.RSI_OVERBOUGHT:
        return "Potentially Overbought"
    if value < settings.RSI_OVERSOLD:
        return "Potentially Oversold"
    return "Neutral"


def macd_state(macd_value: float, signal_value: float) -> str:
    """Simple bullish/bearish technical state based on MACD vs Signal line."""
    if pd.isna(macd_value) or pd.isna(signal_value):
        return "Not enough data"
    return "Bullish (MACD > Signal)" if macd_value > signal_value else "Bearish (MACD < Signal)"


def daily_return(series: pd.Series) -> pd.Series:
    """Percentage daily return."""
    return series.pct_change()


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of daily returns — a simple volatility proxy."""
    return daily_return(series).rolling(window=window, min_periods=window).std()


def momentum(series: pd.Series, window: int = 10) -> pd.Series:
    """Simple momentum: current price minus price `window` periods ago."""
    return series.diff(window)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all required and optional indicators and append them as new
    columns on a copy of the input dataframe. Expects a 'Close' column.
    """
    data = df.copy()
    close = data["Close"]

    for window in settings.SMA_WINDOWS:
        data[f"SMA_{window}"] = sma(close, window)

    for span in settings.EMA_WINDOWS:
        data[f"EMA_{span}"] = ema(close, span)

    macd_df = macd(close)
    data = pd.concat([data, macd_df], axis=1)

    data["RSI_14"] = rsi(close, settings.RSI_PERIOD)

    # Optional extra features (not part of the primary required set, but useful).
    data["Daily_Return"] = daily_return(close)
    data["Volatility_20"] = rolling_volatility(close, 20)
    data["Momentum_10"] = momentum(close, 10)
    if "Volume" in data.columns:
        data["Volume_Change"] = data["Volume"].pct_change()

    return data
