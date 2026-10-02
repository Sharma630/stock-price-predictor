"""Forecasting page: next-step forecast and experimental multi-day forecast."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from config import settings
from src.prediction import generate_forecast, generate_multi_day_forecast, model_exists
from src.ui_helpers import guarded, load_data_with_feedback, render_sidebar, require_selection
from src.utils import format_index_points, format_percentage

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
render_sidebar()
st.title("🔮 Forecasting")

index_key, start_date, end_date, lookback = require_selection()
label = settings.INDEX_OPTIONS[index_key]["label"]
st.caption(f"Index: **{label}**  |  Range: {start_date} → {end_date}")

if not model_exists(index_key):
    st.warning(
        f"No trained model found for {label}. Please train a model on the "
        f"**Model Training** page first."
    )
    st.stop()

raw_df, feature_df = load_data_with_feedback(index_key, start_date, end_date)

st.info(settings.EDUCATIONAL_DISCLAIMER)

if st.button("🎯 GENERATE FORECAST", type="primary"):
    forecast = guarded(generate_forecast, raw_df, index_key, error_prefix="Could not generate forecast")

    st.markdown("### Next-Period Forecast")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", format_index_points(forecast["latest_close"]))
    c2.metric(
        "Predicted Close",
        format_index_points(forecast["predicted_close"]),
        format_percentage(forecast["change_pct"]),
    )
    c3.metric("Expected Change", format_index_points(forecast["change"]))
    direction_emoji = {"UP": "🟢 UP", "DOWN": "🔴 DOWN", "NEUTRAL": "⚪ NEUTRAL"}[forecast["direction"]]
    c4.metric("Predicted Direction", direction_emoji)

    st.caption(
        f"Prediction generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Model trained at: {forecast['model_trained_at']}  |  "
        f"Lookback window: {forecast['lookback']} days  |  "
        f"Latest data date: {forecast['latest_date']}  |  "
        f"Units: {forecast['currency_label']}"
    )

    st.markdown("---")
    st.markdown("### Experimental Multi-Day Forecast")
    st.warning(settings.MULTI_STEP_DISCLAIMER)

    multi = guarded(generate_multi_day_forecast, raw_df, index_key, error_prefix="Could not generate multi-day forecast")
    cols = st.columns(len(multi))
    for col, step in zip(cols, multi):
        with col:
            st.metric(
                f"+{step['horizon_days']} day(s)",
                format_index_points(step["predicted_close"]),
                format_percentage(step["change_pct"]),
            )
            st.caption(f"Direction: {step['direction']}")

st.markdown("---")
st.caption(settings.EDUCATIONAL_DISCLAIMER)
