"""Model Performance page: actual vs predicted, error chart, metrics, residuals."""

from __future__ import annotations

import streamlit as st

from config import settings
from src.evaluation import METRIC_EXPLANATIONS
from src.prediction import evaluate_saved_model_on_test_set, model_exists
from src.ui_helpers import guarded, load_data_with_feedback, render_sidebar, require_selection
from src.visualization import actual_vs_predicted_chart, prediction_error_chart, residual_distribution_chart

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
render_sidebar()
st.title("📈 Model Performance")

index_key, start_date, end_date, lookback = require_selection()
label = settings.INDEX_OPTIONS[index_key]["label"]
st.caption(f"Index: **{label}**  |  Range: {start_date} → {end_date}")

if not model_exists(index_key):
    st.warning(f"No trained model found for {label}. Please train a model on the **Model Training** page first.")
    st.stop()

raw_df, _ = load_data_with_feedback(index_key, start_date, end_date)

result = guarded(
    evaluate_saved_model_on_test_set, raw_df, index_key, error_prefix="Could not evaluate the saved model"
)

st.markdown("### Evaluation Metrics (Held-Out Test Set)")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("MAE", result["metrics"]["MAE"])
m2.metric("RMSE", result["metrics"]["RMSE"])
m3.metric("MAPE (%)", result["metrics"]["MAPE"])
m4.metric("R²", result["metrics"]["R2"])
m5.metric("Directional Accuracy (%)", result["metrics"]["Directional_Accuracy"])

with st.expander("What does each metric mean?", expanded=True):
    for k, v in METRIC_EXPLANATIONS.items():
        st.markdown(f"- **{k}**: {v}")

st.warning(
    "Stock-market prediction is inherently probabilistic. Historical accuracy on "
    "this test set does not guarantee future performance."
)

st.markdown("---")
st.plotly_chart(
    actual_vs_predicted_chart(result["dates"], result["y_true"], result["y_pred"]),
    use_container_width=True,
)
st.plotly_chart(
    prediction_error_chart(result["dates"], result["y_true"], result["y_pred"]),
    use_container_width=True,
)

with st.expander("Residual Distribution (optional)"):
    st.plotly_chart(residual_distribution_chart(result["y_true"], result["y_pred"]), use_container_width=True)

st.markdown("---")
meta = result["metadata"]
st.caption(
    f"Model trained at {meta.get('trained_at')}  |  Lookback: {meta.get('lookback')}  |  "
    f"Epochs run: {meta.get('epochs_run')}  |  Training date range: "
    f"{meta.get('date_range', {}).get('start')} → {meta.get('date_range', {}).get('end')}"
)
st.caption(settings.EDUCATIONAL_DISCLAIMER)
