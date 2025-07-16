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
BUY_THRESHOLD  = 0.0002  # 0.02% threshold for signal
SELL_THRESHOLD = 0.0002

# ─── LOAD MODEL & SCALER ONCE ───────────
_model  = load_model(MODEL_DIR / "baru.h5")
_scaler = load(MODEL_DIR / "scaler.joblib")


def _fetch_binance() -> pd.DataFrame:
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


def _fetch_coincap() -> pd.DataFrame:
    url = "https://api.coincap.io/v2/candles"
    params = {
        "exchange": "binance",
        "interval": "m1",
        "baseId":   "bitcoin",
        "quoteId":  "tether",
        "limit":    TIME_STEP + 50
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["time"], unit='ms')
    df.set_index("Date", inplace=True)
    return df.rename(columns={
        "open":  "Open",
        "high":  "High",
        "low":   "Low",
        "close": "Price",
        "volume":"Vol."
    })


def _fetch_coingecko() -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": 2}
    resp = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=['ts','Open','High','Low','Price'])
    df['Date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('Date', inplace=True)
    df['Vol.'] = 0.0
    return df


def _load_live_data() -> pd.DataFrame:
    # 1) Try Binance
    try:
        df = _fetch_binance()
        if len(df) >= TIME_STEP + 50:
            return df
    except Exception:
        pass

    # 2) Try CoinCap
    try:
        df = _fetch_coincap()
        if len(df) >= TIME_STEP + 50:
            return df
    except Exception:
        pass

    # 3) Try CoinGecko
    try:
        df = _fetch_coingecko()
        if len(df) >= TIME_STEP + 50:
            return df
    except Exception:
        pass

    # 4) Fallback to local CSV
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    return df.rename(columns={
        'Price':'Price',
        'Vol.':  'Vol.',
        'Open':  'Open',
        'High':  'High',
        'Low':   'Low'
    })


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['Vol.'] = df['Vol.'].astype(float)
    df.ta.rsi(close=df['Price'], append=True)
    df.ta.macd(close=df['Price'], append=True)
    df['SMA_15'] = df['Price'].rolling(window=15).mean()
    return df.dropna()


def make_prediction() -> dict:
    df = _load_live_data().tail(TIME_STEP + 50)
    df = df[['Price','Open','High','Low','Vol.']]
    df = _prepare_dataframe(df)
    window = df.tail(TIME_STEP)
    if len(window) < TIME_STEP:
        return {"error": f"Need {TIME_STEP} rows; got {len(window)}"}

    feature_cols = list(_scaler.feature_names_in_)
    window = window[feature_cols]
    scaled = _scaler.transform(window)

    X      = scaled.reshape(1, TIME_STEP, scaled.shape[1])
    pred_s = _model.predict(X, verbose=0)[0,0]
    dummy  = np.zeros((1, scaled.shape[1])); dummy[0,0] = pred_s
    pred_p = _scaler.inverse_transform(dummy)[0,0]

    curr_p = window['Price'].iloc[-1]

    # Trading logic with tight SL/TP around the predicted move
    action, sl, tp = "HOLD", None, None
    buffer = BUY_THRESHOLD  # 0.02%

    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action = "STRONG BUY"
        sl = curr_p * (1 - buffer)         # SL just below entry by 0.02%
        tp = pred_p * (1 - buffer)         # TP just below predicted price

    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action = "STRONG SELL"
        sl = curr_p * (1 + buffer)         # SL just above entry by 0.02%
        tp = pred_p * (1 + buffer)         # TP just above predicted price

    return {
        "time":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":   float(curr_p),
        "predicted_price": float(pred_p),
        "action":          action,
        "stop_loss":       sl,
        "take_profit":     tp
    }
