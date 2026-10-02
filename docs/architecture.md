# System Architecture

## High-Level Pipeline

```
Market Data Source (Yahoo Finance via yfinance)
        |
        v
   OHLCV Data (Open, High, Low, Close, Volume)
        |
        v
   Preprocessing
   (sort, de-duplicate, handle missing values)
        |
        v
   Technical Indicators
   MA (SMA 20/50) | EMA (12/26) | MACD (12,26,9) | RSI (14)
        |
        v
   Feature Matrix (13 columns)
        |
        v
   MinMaxScaler (fit on TRAIN split only)
        |
        v
   Sliding Windows (lookback = 60 trading days)
        |
        v
   LSTM Network
   LSTM(64, return_sequences=True) -> Dropout(0.2)
   -> LSTM(32) -> Dropout(0.2)
   -> Dense(16, relu) -> Dense(1)
        |
        v
   Predictions (next scaled Close)
        |
        +------> Performance Evaluation (MAE, RMSE, MAPE, R², Directional Accuracy)
        |
        v
   Streamlit Dashboard (charts, metrics, forecast cards)
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config/settings.py` | Single source of truth for paths, tickers, indicator parameters, LSTM hyperparameters, split ratios, disclaimers. |
| `src/data_loader.py` | Downloads OHLCV data from Yahoo Finance; falls back to bundled sample data with a clear on-screen notice if live download fails. |
| `src/preprocessing.py` | Cleans raw data (sorting, de-duplication, missing values) and performs the strictly chronological train/validation/test split. |
| `src/indicators.py` | Pure functions implementing SMA, EMA, MACD, RSI, and a few optional indicators (daily return, volatility, momentum). |
| `src/feature_engineering.py` | Builds the final feature dataframe and fits/saves the `MinMaxScaler` on the training split only. |
| `src/sequences.py` | Converts a scaled 2D feature matrix into 3D sliding-window sequences for the LSTM, and extracts the most recent window for forecasting. |
| `src/model.py` | Defines and compiles the LSTM architecture (TensorFlow/Keras imported lazily). |
| `src/training.py` | Orchestrates the full training pipeline end-to-end and persists the model/scaler/metadata bundle. |
| `src/evaluation.py` | Computes MAE, RMSE, MAPE, R², and Directional Accuracy. |
| `src/prediction.py` | Loads a saved model bundle and produces next-step and experimental multi-day forecasts, plus a test-set re-evaluation helper for the Performance page. |
| `src/visualization.py` | Builds all Plotly figures (candlestick, MA overlay, MACD, RSI, actual-vs-predicted, error, residuals). |
| `src/ui_helpers.py` | Streamlit-specific caching and friendly error-handling wrappers shared across pages. |
| `app.py` + `pages/*.py` | The Streamlit multi-page UI. |

## Why This Design

- **Modularity** — each pipeline stage is a small, independently testable
  function/module, which keeps the codebase understandable for a viva and
  avoids duplicated indicator formulas.
- **No data leakage** — splitting happens chronologically before scaling,
  and the scaler is fit only on the training portion.
- **Lazy TensorFlow import** — `src/model.py` and `src/prediction.py` import
  TensorFlow inside functions rather than at module load time, so the rest
  of the pipeline (indicators, preprocessing, sequence generation) can be
  imported and unit-tested even before TensorFlow finishes installing.
- **Explicit training trigger** — training never runs automatically on app
  startup; it only runs when the user presses "TRAIN MODEL" on the Model
  Training page.

## Cross-Checking Indicator Values (TradingView)

The calculated SMA/EMA/MACD/RSI values in this project can be manually
cross-checked against TradingView's public charting tools for the same
ticker and date range, as a sanity check during development or viva
preparation. ChartIQ is not bundled as a dependency because it requires a
commercial license; if needed in the future, it could be integrated through
a small adapter module without changing the rest of the pipeline.
