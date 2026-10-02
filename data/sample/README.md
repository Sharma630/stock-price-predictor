# Bundled Sample Data

`NIFTY50_sample.csv` and `SENSEX_sample.csv` are **synthetically generated
demo datasets** (random-walk OHLCV series with realistic magnitude and
volatility), bundled so the application's charts, indicators, and pipeline
can be demonstrated even without an internet connection.

They are **not real historical market data**. When the app uses this
fallback, it always displays the notice:

> "Using bundled sample dataset."

For real analysis and model training, use the live download path (the
default), which pulls actual historical data for `^NSEI` (NIFTY 50) and
`^BSESN` (SENSEX) from Yahoo Finance via `yfinance`.
