"""Model Training page: configure and trigger LSTM training for the selected index."""

from __future__ import annotations

import streamlit as st

from config import settings
from src.evaluation import METRIC_EXPLANATIONS
from src.prediction import model_exists
from src.training import train_pipeline
from src.ui_helpers import load_data_with_feedback, render_sidebar, require_selection

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")
render_sidebar()
st.title("🧠 Model Training")

index_key, start_date, end_date, lookback = require_selection()
label = settings.INDEX_OPTIONS[index_key]["label"]
st.caption(f"Index: **{label}**  |  Range: {start_date} → {end_date}")

raw_df, feature_df = load_data_with_feedback(index_key, start_date, end_date)

st.markdown("### Training Settings")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.text_input("Ticker", value=settings.INDEX_OPTIONS[index_key]["yf_ticker"], disabled=True)
with c2:
    st.text_input("Date Range", value=f"{start_date} → {end_date}", disabled=True)
with c3:
    lookback = st.number_input("Lookback (days)", min_value=20, max_value=120, value=lookback, step=5)
with c4:
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=st.session_state.get("epochs", settings.DEFAULT_EPOCHS))

batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=st.session_state.get("batch_size", settings.DEFAULT_BATCH_SIZE))

n = len(feature_df)
train_n = int(n * settings.TRAIN_RATIO)
val_n = int(n * (settings.TRAIN_RATIO + settings.VAL_RATIO)) - train_n
test_n = n - train_n - val_n

st.markdown("### Dataset Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Dataset Rows", f"{n:,}")
m2.metric("Training Samples", f"{max(train_n - lookback, 0):,}")
m3.metric("Validation Samples", f"{max(val_n - lookback, 0):,}")
m4.metric("Testing Samples", f"{max(test_n - lookback, 0):,}")
m5.metric("Number of Features", f"{len(settings.FEATURE_COLUMNS)}")

if model_exists(index_key):
    st.success(f"A trained model already exists for {label}. You can use it on the Forecasting page, or retrain below.")
else:
    st.info(f"No trained model exists yet for {label}. Train one below before using the Forecasting page.")

st.markdown("---")

if st.button("🚀 TRAIN MODEL", type="primary"):
    status_box = st.status("Starting training pipeline...", expanded=True)
    progress_bar = st.progress(0.0)
    stages = [
        "Cleaning raw market data...",
        "Calculating technical indicators (SMA, EMA, MACD, RSI)...",
        "Splitting data chronologically and fitting scaler on training data only...",
        "Building sliding-window sequences",
        "Building LSTM model...",
        "Training for up to",
        "Evaluating on the held-out test set...",
        "Saving model, scaler, and metadata...",
        "Training complete.",
    ]

    def progress_callback(msg: str) -> None:
        status_box.write(msg)
        for i, stage in enumerate(stages):
            if stage in msg:
                progress_bar.progress((i + 1) / len(stages))
                break

    try:
        result = train_pipeline(
            raw_df,
            index_key,
            lookback=int(lookback),
            epochs=int(epochs),
            batch_size=int(batch_size),
            progress_callback=progress_callback,
        )
        status_box.update(label="✅ Training complete!", state="complete", expanded=False)
        st.success(f"Model trained and saved for {label}.")

        st.markdown("### Training Results")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("MAE", result.metrics["MAE"])
        r2.metric("RMSE", result.metrics["RMSE"])
        r3.metric("MAPE (%)", result.metrics["MAPE"])
        r4.metric("R²", result.metrics["R2"])
        r5.metric("Directional Accuracy (%)", result.metrics["Directional_Accuracy"])

        with st.expander("What do these metrics mean?"):
            for k, v in METRIC_EXPLANATIONS.items():
                st.markdown(f"- **{k}**: {v}")

        paths = settings.model_paths(index_key)
        st.caption(
            f"Saved to: `{paths['model'].name}`, `{paths['scaler'].name}`, `{paths['metadata'].name}` "
            f"under `models/`."
        )
    except Exception as exc:
        status_box.update(label="❌ Training failed", state="error")
        st.error(f"Training failed: {exc}")

st.markdown("---")
st.caption(settings.EDUCATIONAL_DISCLAIMER)
