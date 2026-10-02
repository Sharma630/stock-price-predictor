# Project Workflow

This document explains, in plain language, how data flows through the
application from user action to final forecast.

## End-to-End Execution Flow

1. **User opens the app** (`streamlit run app.py`) and selects an index
   (NIFTY 50 or SENSEX), a date range, and a lookback window in the sidebar.
2. **Dashboard page** downloads (or loads cached/sample) OHLCV data, cleans
   it, computes all technical indicators, and displays candlestick and
   moving-average charts plus quick RSI/MACD status.
3. **Technical Analysis page** presents dedicated views for Moving
   Averages, MACD, and RSI, with plain-language explanations.
4. **Model Training page** lets the user configure lookback/epochs/batch
   size and press **TRAIN MODEL**. This triggers the full pipeline:
   clean → feature engineer → chronologically split → scale → build
   sequences → train the LSTM → evaluate on the test set → save the model,
   scaler, and metadata to disk under `models/`.
5. **Forecasting page** loads the saved model bundle for the selected
   index, builds the most recent sliding window from current data, and
   produces a next-period prediction (Latest Close, Predicted Close,
   Expected Change, Direction), plus an experimental recursive multi-day
   forecast.
6. **Model Performance page** reconstructs the same chronological test
   split used during training, runs the saved model over it (no
   retraining), and displays Actual-vs-Predicted and error charts along
   with MAE/RMSE/MAPE/R²/Directional Accuracy.
7. **About Project page** presents the full academic write-up for viva and
   documentation purposes.

## Data Flow Diagram (Conceptual)

```
User
  |
  |  selects index / date range / lookback
  v
Forecasting System (Streamlit app)
  |
  |--> Data Source (Yahoo Finance / bundled sample data)
  |         returns OHLCV data
  |
  |--> Technical Indicator Module (src/indicators.py)
  |         returns SMA / EMA / MACD / RSI columns
  |
  |--> LSTM Model (src/model.py, src/training.py, src/prediction.py)
  |         trains on historical sequences OR
  |         loads a saved model and predicts the next value
  v
Forecast Output (Predicted Close, Change %, Direction)
  |
  v
User (views results in the Forecasting / Model Performance pages)
```

## Why Training Is Manual

Training is only triggered by an explicit button press (or, in principle,
a direct call to `train_pipeline`) — never automatically when the app
starts or reruns. This keeps the app responsive and avoids surprising the
user with a long-running CPU training job every time a widget is touched.

## Why the Split Is Chronological

Shuffling time-series data before splitting would let the model "see the
future" during training (data leakage), producing misleadingly good
metrics that would not hold up on genuinely new data. The 70/15/15 split
here always keeps validation and test data strictly after the training
period in time.
