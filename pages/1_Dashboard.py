"""Dashboard page: historical overview, candlestick chart, quick indicator status."""

from __future__ import annotations

import streamlit as st

from config import settings
from src.indicators import macd_state, rsi_interpretation
from src.ui_helpers import load_data_with_feedback, render_sidebar, require_selection
from src.utils import format_index_points, format_percentage
from src.visualization import candlestick_chart, moving_average_chart

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
render_sidebar()
st.title("📊 Dashboard")

index_key, start_date, end_date, lookback = require_selection()
label = settings.INDEX_OPTIONS[index_key]["label"]
st.caption(f"Index: **{label}**  |  Range: {start_date} → {end_date}")

raw_df, feature_df = load_data_with_feedback(index_key, start_date, end_date)

latest = feature_df.iloc[-1]
previous = feature_df.iloc[-2] if len(feature_df) > 1 else latest
pct_change = (latest["Close"] - previous["Close"]) / previous["Close"]

st.markdown(f"**Latest data date:** {latest['Date'].date()}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Open", format_index_points(latest["Open"]))
col2.metric("High", format_index_points(latest["High"]))
col3.metric("Low", format_index_points(latest["Low"]))
col4.metric("Close", format_index_points(latest["Close"]), format_percentage(pct_change))
col5.metric("Volume", f"{int(latest['Volume']):,}")

st.markdown("---")

quick1, quick2 = st.columns(2)
with quick1:
    st.markdown("#### Quick RSI Status")
    st.metric("RSI (14)", f"{latest['RSI_14']:.1f}", rsi_interpretation(latest["RSI_14"]))
with quick2:
    st.markdown("#### Quick MACD Status")
    st.metric("MACD State", macd_state(latest["MACD"], latest["MACD_SIGNAL"]))

st.markdown("---")

st.plotly_chart(candlestick_chart(feature_df, title=f"{label} — Candlestick"), use_container_width=True)
st.plotly_chart(moving_average_chart(feature_df, title=f"{label} — Price with SMA/EMA Overlay"), use_container_width=True)

with st.expander("ℹ️ What am I looking at?"):
    st.markdown(
        """
- **Candlestick chart** — each candle shows the Open, High, Low, and Close
  for one trading day. Green candles closed higher than they opened; red
  candles closed lower.
- **Moving average overlay** — SMA(20/50) and EMA(12/26) smooth out daily
  noise to reveal the underlying trend. When shorter-period averages cross
  above longer-period averages, it is often called a **Golden Cross**
  (bullish); the reverse is a **Death Cross** (bearish). See the Technical
  Analysis page for details.
        """
    )

st.caption(settings.EDUCATIONAL_DISCLAIMER)
