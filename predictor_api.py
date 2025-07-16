# predictor_api.py

import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from joblib import load
from tensorflow.keras.models import load_model
import ccxt
import requests

# ─── PATH SETUP ─────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "Model1min")

# ─── CONSTANTS ──────────────────────────
TIME_STEP      = 60
BUY_THRESHOLD  = 0.0005
SELL_THRESHOLD = 0.0005
SL_PCT         = 0.001   # 0.1% stop‐loss
TP_PCT         = 0.005   # 0.5% take‐profit

# ─── LOAD MODEL & SCALER ONCE ───────────
_model  = load_model(os.path.join(MODEL_DIR, "baru.h5"))
_scaler = load(os.path.join(MODEL_DIR, "scaler.joblib"))

def _load_live_data() -> pd.DataFrame:
    """
    Try Binance first; if unavailable, fall back to Coingecko OHLC.
    """
    try:
        ex   = ccxt.binance()
        bars = ex.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=TIME_STEP + 50)
        df   = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume'])
    except Exception as e:
        # 451 or other error → use Coingecko's free OHLC endpoint
        url = (
          "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
          f"?vs_currency=usd&days=2"
        )
        data = requests.get(url).json()  # returns [ [time, o,h,l,c], ... ]
        df   = pd.DataFrame(data[-(TIME_STEP+50):], columns=['ts','open','high','low','close'])
        df['volume'] = 0  # Coingecko OHLC has no volume
    df['Date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('Date', inplace=True)
    return df.rename(columns={
        'close':'Price',
        'volume':'Vol.',
        'open':'Open',
        'high':'High',
        'low':'Low'
    })

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and drop NaNs.
    Expects df with columns ['Price','Open','High','Low','Vol.'].
    """
    df['Vol.'] = df['Vol.'].astype(float)
    df.ta.rsi(close=df['Price'], append=True)
    df.ta.macd(close=df['Price'], append=True)
    df['SMA_15'] = df['Price'].rolling(window=15).mean()
    return df.dropna()

def make_prediction() -> dict:
    """
    Fetches live data, prepares it, runs the model, and returns
    a dict with time, current_price, predicted_price, action, SL, TP.
    """
    # 1. Fetch & slice live data
    df = _load_live_data().tail(TIME_STEP + 50)

    # 2. Keep only the base features
    df = df[['Price','Open','High','Low','Vol.']]

    # 3. Compute indicators & drop NaN
    df_prepped = _prepare_dataframe(df)
    window     = df_prepped.tail(TIME_STEP)
    if len(window) < TIME_STEP:
        return {"error": f"Need {TIME_STEP} rows; got {len(window)}"}

    # 4. Align to scaler’s training features and scale
    feature_cols   = list(_scaler.feature_names_in_)
    window_aligned = window[feature_cols]
    scaled         = _scaler.transform(window_aligned)
    X              = scaled.reshape(1, TIME_STEP, scaled.shape[1])

    # 5. Predict
    pred_s = _model.predict(X, verbose=0)[0, 0]
    dummy  = np.zeros((1, scaled.shape[1])); dummy[0, 0] = pred_s
    pred_p = _scaler.inverse_transform(dummy)[0, 0]

    # 6. Current price
    curr_p = window['Price'].iloc[-1]

    # 7. Trading decision
    action, sl, tp = "HOLD", None, None
    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action = "STRONG BUY"
        sl     = curr_p * (1 - SL_PCT)
        tp     = curr_p * (1 + TP_PCT)
    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action = "STRONG SELL"
        sl     = curr_p * (1 + SL_PCT)
        tp     = curr_p * (1 - TP_PCT)

    # 8. Return results
    return {
        "time":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":  float(curr_p),
        "predicted_price": float(pred_p),
        "action":          action,
        "stop_loss":       sl,
        "take_profit":     tp
    }
