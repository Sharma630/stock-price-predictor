"""Technical Analysis page: dedicated sections for Moving Averages, MACD, and RSI."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src.indicators import macd_state, rsi_interpretation
from src.ui_helpers import load_data_with_feedback, render_sidebar, require_selection
from src.visualization import macd_chart, moving_average_chart, rsi_chart

st.set_page_config(page_title="Technical Analysis", page_icon="🧮", layout="wide")
render_sidebar()
st.title("🧮 Technical Analysis")

index_key, start_date, end_date, lookback = require_selection()
label = settings.INDEX_OPTIONS[index_key]["label"]
st.caption(f"Index: **{label}**  |  Range: {start_date} → {end_date}")

raw_df, feature_df = load_data_with_feedback(index_key, start_date, end_date)
latest = feature_df.iloc[-1]

tab_ma, tab_macd, tab_rsi = st.tabs(["Moving Averages", "MACD", "RSI"])

with tab_ma:
    st.markdown("### Moving Averages")
    with st.expander("What are SMA and EMA?", expanded=False):
        st.markdown(
            """
- **SMA (Simple Moving Average)** — the plain average of the closing price
  over the last *N* days. It reacts slowly and smoothly to price changes.
- **EMA (Exponential Moving Average)** — a weighted average that gives more
  importance to recent prices, so it reacts faster than an SMA of the same
  length.
- **Golden Cross** — a shorter-period average (e.g. SMA 20) crosses **above**
  a longer-period average (e.g. SMA 50); often interpreted as a bullish
  signal.
- **Death Cross** — the shorter-period average crosses **below** the longer
  one; often interpreted as a bearish signal.

These are lagging indicators derived from past prices — they describe the
trend that has already happened, not a guarantee of what happens next.
            """
        )
    st.plotly_chart(moving_average_chart(feature_df), use_container_width=True)

    # Highlight the most recent SMA20/SMA50 crossover, if one exists in view.
    cross_series = (feature_df["SMA_20"] - feature_df["SMA_50"]).dropna()
    if len(cross_series) > 1:
        sign_changes = (cross_series > 0).astype(int).diff().fillna(0)
        crossover_points = feature_df.loc[sign_changes[sign_changes != 0].index]
        if not crossover_points.empty:
            last_cross = crossover_points.iloc[-1]
            cross_type = "Golden Cross (bullish)" if last_cross["SMA_20"] > last_cross["SMA_50"] else "Death Cross (bearish)"
            st.info(f"Most recent SMA 20/50 crossover: **{cross_type}** on {last_cross['Date'].date()}.")

with tab_macd:
    st.markdown("### MACD (Moving Average Convergence Divergence)")
    with st.expander("How is MACD calculated?", expanded=False):
        st.markdown(
            """
- **MACD line** = EMA(12) − EMA(26)
- **Signal line** = EMA(MACD line, 9)
- **Histogram** = MACD line − Signal line

**Bullish** when MACD > Signal; **Bearish** when MACD < Signal. This is a
descriptive technical state, not a guaranteed buy/sell recommendation.
            """
        )
    st.plotly_chart(macd_chart(feature_df), use_container_width=True)
    st.metric("Current MACD State", macd_state(latest["MACD"], latest["MACD_SIGNAL"]))

with tab_rsi:
    st.markdown("### RSI (Relative Strength Index)")
    with st.expander("How is RSI calculated?", expanded=False):
        st.markdown(
            f"""
RSI measures the speed and magnitude of recent price changes on a 0–100
scale using a {settings.RSI_PERIOD}-day period.

- **RSI > {settings.RSI_OVERBOUGHT}** → Potentially Overbought
- **RSI < {settings.RSI_OVERSOLD}** → Potentially Oversold
- Otherwise → Neutral

These thresholds are heuristics used widely in technical analysis, not
guaranteed buy/sell signals.
            """
        )
    st.plotly_chart(rsi_chart(feature_df), use_container_width=True)
    st.metric("Current RSI (14)", f"{latest['RSI_14']:.1f}", rsi_interpretation(latest["RSI_14"]))

st.markdown("---")
st.caption(settings.EDUCATIONAL_DISCLAIMER)
