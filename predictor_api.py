# predictor_api.py

import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from joblib import load
from tensorflow.keras.models import load_model
import ccxt
from pathlib import Path

# ─── PATH SETUP ─────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "Model1min")
CSV_PATH   = Path(BASE_DIR) / "data" / "BTC_DATA_V3.0.csv"

# ─── CONSTANTS ──────────────────────────
TIME_STEP      = 60
BUY_THRESHOLD  = 0.0005
SELL_THRESHOLD = 0.0005
SL_PCT         = 0.001   # 0.1% stop-loss
TP_PCT         = 0.005   # 0.5% take-profit

# ─── LOAD MODEL & SCALER ONCE ───────────
_model  = load_model(os.path.join(MODEL_DIR, "baru.h5"))
_scaler = load(os.path.join(MODEL_DIR, "scaler.joblib"))

def _load_live_data() -> pd.DataFrame:
    """
    Try to fetch 1m OHLCV from Binance (limit=TIME_STEP+50).
    If that fails or returns too few rows, fall back to the local CSV.
    """
    try:
        exchange = ccxt.binance()
        bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=TIME_STEP + 50)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df.rename(columns={
            'close': 'Price',
            'volume': 'Vol.',
            'open': 'Open',
            'high': 'High',
            'low': 'Low'
        })

        # If Binance returned too few rows, force fallback
        if len(df) < TIME_STEP + 50:
            raise ValueError("Binance returned too few rows")

        return df

    except Exception:
        # Fallback to local CSV
        df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        # Ensure our updater wrote the same column names
        df = df.rename(columns={
            'Price': 'Price',
            'Vol.': 'Vol.',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low'
        })
        return df

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, and SMA_15 on the 'Price' column,
    drop any rows with NaNs from indicator calculations.
    """
    df['Vol.'] = df['Vol.'].astype(float)
    df.ta.rsi(close=df['Price'], append=True)
    df.ta.macd(close=df['Price'], append=True)
    df['SMA_15'] = df['Price'].rolling(window=15).mean()
    return df.dropna()

def make_prediction() -> dict:
    """
    Fetch & prepare live data, run the model, and return:
    {
      time, current_price, predicted_price, action,
      stop_loss, take_profit
    }
    """
    # 1) Load & slice
    df = _load_live_data().tail(TIME_STEP + 50)

    # 2) Keep base features
    df = df[['Price', 'Open', 'High', 'Low', 'Vol.']]

    # 3) Compute indicators & drop NaNs
    df_prepped = _prepare_dataframe(df)
    window     = df_prepped.tail(TIME_STEP)
    if len(window) < TIME_STEP:
        return {"error": f"Need {TIME_STEP} rows; got {len(window)}"}

    # 4) Align to scaler feature order & scale
    feature_cols   = list(_scaler.feature_names_in_)
    window_aligned = window[feature_cols]
    scaled         = _scaler.transform(window_aligned)

    # 5) Predict
    X     = scaled.reshape(1, TIME_STEP, scaled.shape[1])
    pred_s = _model.predict(X, verbose=0)[0, 0]

    # 6) Inverse-scale price
    dummy  = np.zeros((1, scaled.shape[1]))
    dummy[0, 0] = pred_s
    pred_p = _scaler.inverse_transform(dummy)[0, 0]

    # 7) Current price
    curr_p = window['Price'].iloc[-1]

    # 8) Trading logic
    action, sl, tp = "HOLD", None, None
    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action = "STRONG BUY"
        sl     = curr_p * (1 - SL_PCT)
        tp     = curr_p * (1 + TP_PCT)
    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action = "STRONG SELL"
        sl     = curr_p * (1 + SL_PCT)
        tp     = curr_p * (1 - TP_PCT)

    # 9) Return the full result
    return {
        "time":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":   float(curr_p),
        "predicted_price": float(pred_p),
        "action":          action,
        "stop_loss":       sl,
        "take_profit":     tp
    }
