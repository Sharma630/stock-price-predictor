# Viva Questions & Answers

## Fundamentals

**1. Why is stock market forecasting considered difficult?**
Prices are influenced by a huge number of factors (macroeconomics, news,
sentiment, policy, global markets) that are not all captured in historical
price data, and markets react quickly to new information, making patterns
noisy and short-lived.

**2. What is non-stationarity, and why does it matter here?**
A non-stationary series has statistical properties (mean, variance) that
change over time. Stock prices are non-stationary, which is why we use
returns, indicators, and normalization rather than modeling raw prices in
isolation.

**3. What is time-series data?**
Data points indexed in time order, where the sequence and spacing of
observations matters — unlike i.i.d. tabular data, past values can inform
future ones.

**4. What does OHLCV stand for?**
Open, High, Low, Close, Volume — the standard summary of a security's
trading activity for a given period (here, one trading day).

## Technical Indicators

**5. What is SMA?**
Simple Moving Average — the arithmetic mean of the closing price over the
last N periods.

**6. What is EMA, and how does it differ from SMA?**
Exponential Moving Average — a weighted average that gives more weight to
recent prices, so it reacts faster to new price changes than an SMA of the
same length.

**7. What is a Golden Cross / Death Cross?**
A Golden Cross is when a shorter-period average crosses above a
longer-period average (often seen as bullish); a Death Cross is the
opposite (often seen as bearish).

**8. What is MACD and how is it computed?**
Moving Average Convergence Divergence. MACD = EMA(12) − EMA(26); Signal =
EMA(MACD, 9); Histogram = MACD − Signal.

**9. What does the MACD histogram tell us?**
The distance between the MACD line and its signal line — widening
histogram bars suggest strengthening momentum in the current direction.

**10. What is RSI, and what range does it take?**
Relative Strength Index — a momentum oscillator ranging roughly 0–100,
based on the ratio of average gains to average losses over a period
(14 days here).

**11. What do RSI > 70 and RSI < 30 typically indicate?**
Potentially overbought and potentially oversold conditions, respectively —
heuristics, not guarantees.

**12. Why do indicators need a "warm-up" period?**
Rolling/EMA-based calculations need a minimum number of prior observations
before they produce a defined value, so the first N rows are NaN and must
be dropped.

## Data Preparation

**13. What is MinMaxScaler, and why use it here?**
It rescales features to a fixed range (here, [0, 1]). LSTMs train more
reliably when input features are on a similar, bounded scale.

**14. What is data leakage, and how is it avoided in this project?**
Data leakage happens when information from the future (or from
validation/test data) influences training. Here it is avoided by (a)
splitting chronologically before scaling and (b) fitting the scaler only
on the training split.

**15. Why is the data split chronologically instead of randomly?**
Randomly shuffling time-series data would let the model train on data that
comes chronologically after some of its test data, leaking future
information and producing misleadingly optimistic metrics.

**16. What train/validation/test ratio is used?**
70% / 15% / 15%, applied in chronological order (train earliest, test
latest).

## Sequence Modeling & LSTM

**17. What is a lookback window?**
The number of past time steps (trading days) used as one input sequence
for the model — 60 by default in this project.

**18. What is an LSTM, and why use it instead of a plain RNN?**
Long Short-Term Memory — a recurrent neural network variant with gating
mechanisms that help it retain information over longer sequences and
mitigate the vanishing gradient problem that hurts plain RNNs.

**19. What is the vanishing gradient problem?**
During backpropagation through many time steps, gradients can shrink
exponentially, making it hard for a plain RNN to learn long-range
dependencies. LSTM's gating mechanisms help preserve gradient flow.

**20. Why use Dropout in this model?**
Dropout randomly disables a fraction of neurons during training, which
helps prevent overfitting by discouraging co-dependence between specific
neurons.

**21. What loss function and optimizer are used, and why?**
Mean Squared Error (MSE) loss, since this is a regression problem
(predicting a continuous closing value), optimized with Adam, a widely
used adaptive-learning-rate optimizer that converges well in practice.

**22. What is Early Stopping, and why is it used?**
It stops training when validation loss stops improving for a set number of
epochs, preventing overfitting and saving training time.

**23. What is overfitting, and how is it addressed here?**
When a model learns patterns specific to the training data that do not
generalize. Addressed here via Dropout, Early Stopping, and evaluating on
a held-out chronological test set.

## Evaluation Metrics

**24. What does MAE measure?**
Mean Absolute Error — the average absolute difference between actual and
predicted values, in the same units as the target.

**25. What does RMSE measure, and how does it differ from MAE?**
Root Mean Squared Error — like MAE, but squares errors before averaging
(then takes the square root), so it penalizes larger errors more heavily.

**26. What does R² (R-squared) indicate?**
The proportion of variance in the target variable that is explained by the
model's predictions; closer to 1 is better, 0 means no better than
predicting the mean.

**27. What is Directional Accuracy, and why is it reported separately?**
The percentage of instances where the predicted UP/DOWN movement matches
the actual movement. It is reported separately because a model can have
reasonable price-error metrics while still getting the direction wrong
often (or vice versa), and direction is often more actionable than the
exact price.

**28. Why might a model have low MAE but low Directional Accuracy?**
If the model's predictions closely track the previous price (a common
LSTM behavior on noisy series), price-level errors can look small, while
the day-to-day direction of change may still be effectively a coin flip.

## Limitations & Future Scope

**29. Why can't this project guarantee accurate future predictions?**
Markets are influenced by information not captured in OHLCV data
(news, macroeconomic shifts, sentiment), and historical patterns do not
always repeat; deep learning models here are probabilistic estimators, not
oracles.

**30. What is the difference between price prediction and directional
prediction?**
Price prediction estimates the actual next closing value; directional
prediction only classifies whether the price is expected to go UP, DOWN,
or stay roughly the same (NEUTRAL) — a coarser but sometimes more robust
signal.

**31. How could sentiment analysis improve this project?**
By incorporating signals from news or social media as additional model
inputs, capturing information that pure price/indicator data misses —
listed as future scope in this project.

**32. Why are Transformer models mentioned as future scope instead of
being used now?**
Transformers (e.g. Temporal Fusion Transformer) can model long-range
dependencies and multiple covariates effectively, but add complexity and
computational cost beyond what is appropriate for this academic-scope,
CPU-friendly project.

**33. Why isn't ChartIQ integrated directly?**
ChartIQ is a proprietary charting SDK that typically requires a commercial
license; this project uses Plotly for all visualization and documents how
ChartIQ could be added later through an adapter if a license were
available.

**34. What are the main limitations of this project?**
Indicators are lagging transformations of past prices; the model does not
see macroeconomic or sentiment data; recursive multi-step forecasts
accumulate error; and historical accuracy does not guarantee future
performance.
