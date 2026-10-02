"""
Reusable Plotly chart builders for the Streamlit dashboard: candlestick,
closing price, moving-average overlays, MACD, RSI, and actual-vs-predicted.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config import settings


def candlestick_chart(df: pd.DataFrame, title: str = "Price Candlestick") -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Index Points",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=450,
    )
    return fig


def closing_price_chart(df: pd.DataFrame, title: str = "Closing Price") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Index Points", template="plotly_white", height=400)
    return fig


def moving_average_chart(df: pd.DataFrame, title: str = "Price with Moving Averages") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(width=1.5)))
    for window in settings.SMA_WINDOWS:
        col = f"SMA_{window}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))
    for span in settings.EMA_WINDOWS:
        col = f"EMA_{span}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col, line=dict(dash="dot")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Index Points", template="plotly_white", height=450)
    return fig


def macd_chart(df: pd.DataFrame, title: str = "MACD") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], mode="lines", name="Signal"))
    colors = ["green" if v >= 0 else "red" for v in df["MACD_HIST"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="Histogram", marker_color=colors, opacity=0.5))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="MACD", template="plotly_white", height=350)
    return fig


def rsi_chart(df: pd.DataFrame, title: str = "RSI (14)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], mode="lines", name="RSI 14"))
    fig.add_hline(y=settings.RSI_OVERBOUGHT, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=settings.RSI_OVERSOLD, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100], template="plotly_white", height=350
    )
    return fig


def actual_vs_predicted_chart(dates, y_true, y_pred, title: str = "Actual vs Predicted (Test Set)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="Predicted", line=dict(dash="dot")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Index Points", template="plotly_white", height=450)
    return fig


def prediction_error_chart(dates, y_true, y_pred, title: str = "Prediction Error") -> go.Figure:
    import numpy as np

    errors = np.asarray(y_true) - np.asarray(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=errors, name="Error (Actual - Predicted)"))
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Error", template="plotly_white", height=350)
    return fig


def residual_distribution_chart(y_true, y_pred, title: str = "Residual Distribution") -> go.Figure:
    import numpy as np

    errors = np.asarray(y_true) - np.asarray(y_pred)
    fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=30)])
    fig.update_layout(title=title, xaxis_title="Residual (Actual - Predicted)", yaxis_title="Count", template="plotly_white", height=350)
    return fig
