"""
Stock Market Index Movement Forecasting Using Multiple Indicators and Deep Learning
Main Streamlit entry point.

Run with:
    streamlit run app.py

The sidebar (index / date-range / lookback selectors, shared across all
pages via st.session_state) is rendered by src.ui_helpers.render_sidebar(),
which every page calls independently — Streamlit runs each page as its own
script, so initialization cannot live only here.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from src.ui_helpers import render_sidebar
from src.utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Stock Index Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_home() -> None:
    st.title("Stock Market Index Movement Forecasting")
    st.subheader("Using Multiple Technical Indicators & Deep Learning")

    st.info(settings.EDUCATIONAL_DISCLAIMER)

    st.markdown(
        """
This application forecasts the next-period movement of major Indian stock
market indices (**NIFTY 50** and **S&P BSE SENSEX**) using a combination of
classic technical indicators (Moving Averages, MACD, RSI) and an LSTM
deep-learning model trained on multivariate historical OHLCV sequences.

### How to use this app
1. Choose an index, date range, and lookback window in the sidebar.
2. Open **Dashboard** to explore the historical price action.
3. Open **Technical Analysis** to inspect indicator-based signals.
4. Open **Model Training** to train (or retrain) the LSTM model for the
   selected index — this only happens when you press the Train button.
5. Open **Forecasting** to generate a next-period prediction and an
   experimental multi-day forecast.
6. Open **Model Performance** to review evaluation metrics in detail.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🗂️ Data")
        st.write("Historical OHLCV data via Yahoo Finance, with a bundled sample-data fallback.")
    with col2:
        st.markdown("#### 🧮 Indicators")
        st.write("SMA(20/50), EMA(12/26), MACD(12,26,9), RSI(14), plus optional volatility & momentum features.")
    with col3:
        st.markdown("#### 🧠 Model")
        st.write("A 2-layer LSTM network trained on 60-day multivariate sequences, with dropout regularization.")

    st.markdown("---")
    st.caption(
        "Built as a B.Tech final-year academic project. "
        "Source code is organized under `src/` (pipeline logic) and `pages/` (UI)."
    )


def main() -> None:
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
