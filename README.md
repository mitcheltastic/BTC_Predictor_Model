# BTC 1-Hour Price Predictor

**Live** Streamlit dashboard forecasting Bitcoin’s price 1 hour out.
- Fetches real-time OHLCV via CCXT.
- Computes RSI, MACD, SMA, scales & predicts with your Keras model.
- Displays current vs predicted price plus SL/TP levels.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
