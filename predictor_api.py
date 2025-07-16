# predictor_api.py

import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from joblib import load
from tensorflow.keras.models import load_model
import requests

# ─── PATH SETUP ─────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model1min"
CSV_PATH  = BASE_DIR / "data" / "BTC_DATA_V3.0.csv"

# ─── CONSTANTS ──────────────────────────
TIME_STEP      = 60
BUY_THRESHOLD  = 0.0002  # 0.02% threshold
SELL_THRESHOLD = 0.0002
SL_PCT         = 0.001   # 0.1% stop-loss
TP_PCT         = 0.005   # 0.5% take-profit

# ─── LOAD MODEL & SCALER ONCE ───────────
_model  = load_model(MODEL_DIR / "baru.h5")
_scaler = load(MODEL_DIR / "scaler.joblib")


def _fetch_binance() -> pd.DataFrame:
    """
    Fetch latest 1m OHLCV from Binance.
    """
    import ccxt
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=TIME_STEP + 50)
    df = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume'])
    df['Date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('Date', inplace=True)
    return df.rename(columns={
        'close':'Price',
        'volume':'Vol.',
        'open':'Open',
        'high':'High',
        'low':'Low'
    })


def _fetch_coingecko() -> pd.DataFrame:
    """
    Fetch 1m OHLC for past 2 days from CoinGecko (no volume).
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": 2}  # <-- changed to 2 days
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=['ts','Open','High','Low','Price'])
    df['Date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('Date', inplace=True)
    df['Vol.'] = 0.0
    return df


def _load_live_data() -> pd.DataFrame:
    """
    Attempt Binance → CoinGecko → local CSV fallback.
    Guarantees at least TIME_STEP+50 rows.
    """
    # 1) Binance
    try:
        df = _fetch_binance()
        if len(df) >= TIME_STEP + 50:
            return df
    except Exception:
        pass

    # 2) CoinGecko
    try:
        df = _fetch_coingecko()
        if len(df) >= TIME_STEP + 50:
            return df
    except Exception:
        pass

    # 3) Local CSV fallback
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    return df.rename(columns={
        'Price':'Price', 'Vol.':'Vol.', 'Open':'Open', 'High':'High', 'Low':'Low'
    })


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['Vol.'] = df['Vol.'].astype(float)
    df.ta.rsi(close=df['Price'], append=True)
    df.ta.macd(close=df['Price'], append=True)
    df['SMA_15'] = df['Price'].rolling(window=15).mean()
    return df.dropna()


def make_prediction() -> dict:
    """
    Fetch & slice live data, prepare inputs, run the model,
    and return a summary dict with time, prices, action, SL, TP.
    """
    # 1) Load live data
    df = _load_live_data().tail(TIME_STEP + 50)

    # 2) Keep base features
    df = df[['Price','Open','High','Low','Vol.']]

    # 3) Indicators & cleaning
    df = _prepare_dataframe(df)
    window = df.tail(TIME_STEP)
    if len(window) < TIME_STEP:
        return {"error": f"Need {TIME_STEP} rows; got {len(window)}"}

    # 4) Align to scaler features & scale
    feature_cols = list(_scaler.feature_names_in_)
    window = window[feature_cols]
    scaled = _scaler.transform(window)

    # 5) Predict & inverse-scale
    X      = scaled.reshape(1, TIME_STEP, scaled.shape[1])
    pred_s = _model.predict(X, verbose=0)[0,0]
    dummy  = np.zeros((1, scaled.shape[1])); dummy[0,0] = pred_s
    pred_p = _scaler.inverse_transform(dummy)[0,0]

    # 6) Current price
    curr_p = window['Price'].iloc[-1]

    # 7) Trading logic
    action, sl, tp = "HOLD", None, None
    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action, sl, tp = "STRONG BUY", curr_p*(1-SL_PCT), curr_p*(1+TP_PCT)
    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action, sl, tp = "STRONG SELL", curr_p*(1+SL_PCT), curr_p*(1-TP_PCT)

    # 8) Return result
    return {
        "time":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":   float(curr_p),
        "predicted_price": float(pred_p),
        "action":          action,
        "stop_loss":       sl,
        "take_profit":     tp
    }
