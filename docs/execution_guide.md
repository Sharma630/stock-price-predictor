# Execution Guide

Detailed, step-by-step instructions for running, training, and validating
this project on Windows using VS Code. See the README for the condensed
quick-start version.

## 1. Prerequisites

- Windows 10/11
- Python 3.10 or 3.11 installed, with "Add Python to PATH" checked during
  installation ([python.org/downloads](https://www.python.org/downloads/))
- Visual Studio Code (optional but recommended)
- No paid APIs, API keys, or GPU are required.

## 2. First-Time Setup

1. Extract the project ZIP anywhere on your computer (e.g. `Documents\`).
2. Open the extracted folder in VS Code (`File > Open Folder...`).
3. Open a terminal in VS Code (`` Ctrl+` ``) — it should default to the
   project folder.
4. Run:
   ```
   setup.bat
   ```
   This creates a virtual environment (`.venv`), upgrades pip, and installs
   every package listed in `requirements.txt`. This step can take a few
   minutes, especially the TensorFlow install.

## 3. Running the App

Every time after the first setup, just run:
```
run.bat
```
or, from PowerShell:
```
.\run.ps1
```
Streamlit will print a local URL — open **http://localhost:8501** in your
browser if it does not open automatically.

## 4. Using the App

1. In the sidebar, choose an index (NIFTY 50 or SENSEX), a start/end date,
   and a lookback window.
2. Visit **Dashboard** to see the historical candlestick chart and moving
   averages.
3. Visit **Technical Analysis** to inspect MACD and RSI in detail.
4. Visit **Model Training**, review the dataset overview, and press
   **TRAIN MODEL**. Training runs on CPU and uses a reduced default epoch
   count so it finishes in a reasonable time on an ordinary laptop.
5. Once training finishes, visit **Forecasting** and press
   **GENERATE FORECAST** to see the next-period prediction and an
   experimental multi-day forecast.
6. Visit **Model Performance** to review MAE/RMSE/MAPE/R²/Directional
   Accuracy and the Actual-vs-Predicted chart.

## 5. Running the Automated Tests

```
.venv\Scripts\activate
pytest
```
All tests under `tests/` should pass. They cover indicator correctness
(SMA/EMA/MACD/RSI), preprocessing (cleaning, chronological splitting), and
sequence generation (shape and target alignment).

## 6. Restarting Without Retraining

Once a model has been trained for an index, its files are saved under
`models/trained/`, `models/scalers/`, and `models/metadata/` using
ticker-specific filenames (e.g. `NIFTY50_lstm.keras`). Restarting the app
(`run.bat` again) will detect the existing model automatically — the
Forecasting and Model Performance pages will work immediately without
requiring retraining, and the Model Training page offers a **Retrain**
option if you want to update it.

## 7. Cross-Checking Indicator Values (Optional)

To sanity-check the calculated SMA/EMA/MACD/RSI values, open the same
ticker and date range on TradingView (a free public charting tool) and
compare the indicator values visually. Minor differences can occur due to
small variations in data vendor adjustments or indicator smoothing
conventions; the underlying formulas used in this project follow the
standard, widely documented definitions (see `docs/architecture.md`).

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `python` not recognized | Python not on PATH | Reinstall Python and check "Add Python to PATH" |
| `pip install` fails on `tensorflow-cpu` | Old pip / unsupported Python version | Run `python -m pip install --upgrade pip`, ensure Python 3.10/3.11 |
| "Yahoo Finance returned no data" | No internet, invalid date range, or temporary outage | The app automatically falls back to bundled sample data; try again later for live data |
| "No trained model found" on Forecasting page | Model not trained yet | Go to Model Training and press TRAIN MODEL |
| App looks stuck during training | Normal — training is CPU-bound | Wait for the progress panel to complete; reduce epochs for a faster demo |
| Port 8501 already in use | Another Streamlit instance is running | Close the other instance, or run `streamlit run app.py --server.port 8502` |

## 9. Moving the Project to Another Computer

The project uses only relative paths (via `pathlib`), so it can be copied
to any folder on any Windows computer and will continue to work — just
re-run `setup.bat` on the new machine to rebuild the virtual environment.
