"""About Project page: synopsis-style project write-up for viva/documentation."""

from __future__ import annotations

import streamlit as st

from config import settings
from src.ui_helpers import render_sidebar

st.set_page_config(page_title="About Project", page_icon="ℹ️", layout="wide")
render_sidebar()

st.title("Stock Market Index Movement Forecasting Using Multiple Indicators and Deep Learning")

st.markdown(
    """
### Project Description
This project forecasts the near-term movement of major Indian stock market
indices — **NIFTY 50** and **S&P BSE SENSEX** — by combining classical
technical-analysis indicators with a deep-learning sequence model (LSTM).
It is built as a complete, runnable B.Tech final-year academic project.

### Problem Statement
Stock market indices exhibit noisy, non-stationary, and highly non-linear
behavior that is difficult to model with simple statistical methods alone.
This project investigates whether combining multiple technical indicators
as engineered features with an LSTM network can produce a useful next-period
price and directional (UP/DOWN) forecast for research and educational
purposes.

### Objectives
- Build an end-to-end pipeline: data acquisition → cleaning → feature
  engineering → scaling → sequence modeling → training → evaluation →
  forecasting → visualization.
- Combine SMA/EMA, MACD, and RSI with raw OHLCV data as model inputs.
- Train an LSTM to forecast the next closing value and derive a directional
  signal from it.
- Present results through an interactive, easy-to-understand dashboard.

### Methodology
1. **Data Acquisition** — Historical OHLCV data is downloaded via
   `yfinance` for `^NSEI` (NIFTY 50) and `^BSESN` (SENSEX), with a bundled
   sample dataset as an offline fallback.
2. **Preprocessing** — Chronological sorting, de-duplication, missing-value
   handling, and a strictly chronological train/validation/test split
   (70/15/15) to avoid data leakage.
3. **Feature Engineering** — SMA(20, 50), EMA(12, 26), MACD(12, 26, 9),
   RSI(14), plus optional daily return, volatility, and momentum features.
4. **Scaling** — `MinMaxScaler` fit only on the training split.
5. **Sequence Generation** — Sliding windows of `lookback` (default 60)
   trading days become the LSTM's input tensors.
6. **Modeling** — A 2-layer LSTM network with dropout regularization,
   trained with Adam and MSE loss, using early stopping.
7. **Evaluation** — MAE, RMSE, MAPE, R², and Directional Accuracy on the
   held-out test set.
8. **Forecasting** — Next-step and experimental recursive multi-day
   forecasts, each classified as UP / DOWN / NEUTRAL.

### Technical Indicators Implemented
| Indicator | Purpose |
|---|---|
| SMA (20, 50) | Smooths price to reveal medium-term trend |
| EMA (12, 26) | Trend-following average, more reactive than SMA |
| MACD (12, 26, 9) | Trend momentum via EMA convergence/divergence |
| RSI (14) | Overbought/oversold momentum oscillator (0–100) |

### Deep Learning Explanation
An LSTM (Long Short-Term Memory) network is a recurrent neural network
variant designed to learn patterns across sequences while mitigating the
vanishing-gradient problem that affects plain RNNs on long sequences. Here,
each training example is a 60-day window of 13 engineered features; the
network learns to map that window to the next day's (scaled) closing value.

### Technology Stack
- **Language:** Python 3.10/3.11
- **Web App:** Streamlit
- **Data:** pandas, numpy, yfinance
- **ML/DL:** scikit-learn (MinMaxScaler, metrics), TensorFlow/Keras (LSTM)
- **Visualization:** Plotly

### System Architecture
See `docs/architecture.md` for the full diagram. In short:

```
Market Data → Preprocessing → Technical Indicators → Feature Matrix
→ MinMaxScaler → Sliding Windows → LSTM → Predictions
→ Evaluation + Streamlit Dashboard
```

### Limitations
- Market prices are noisy and influenced by factors outside OHLCV data
  (macroeconomic news, policy changes, geopolitical events).
- Historical performance does not guarantee future performance.
- Technical indicators are lagging transformations of past prices.
- LSTM output is probabilistic/approximate, not a certainty.
- Recursive multi-step forecasts accumulate error with each additional step.

### Future Scope
- News and social-media sentiment analysis as additional model inputs.
- Transformer-based sequence models (e.g. Temporal Fusion Transformer).
- Real-time streaming data ingestion.
- Brokerage API / paper-trading integration (research only).
- Explainable AI (e.g. SHAP) for feature-importance analysis.
- Attention-weight visualization for interpretability.
- Professional charting validation against TradingView (see
  `docs/execution_guide.md` for how calculated indicator values can be
  cross-checked). ChartIQ integration would require a licensed SDK and is
  left as a future adapter rather than a bundled dependency.
"""
)

st.warning(settings.EDUCATIONAL_DISCLAIMER)
