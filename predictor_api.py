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


def _fetch_binance_rest() -> pd.DataFrame:
    """
    Fetch latest 1m OHLCV from Binance REST API (no CCXT).
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": TIME_STEP + 50}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    cols = ["open_time","Open","High","Low","Close","Vol.",
            "close_time","quote_asset_vol","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["Date"] = pd.to_datetime(df["open_time"], unit='ms')
    df.set_index("Date", inplace=True)
    df = df.rename(columns={"Close":"Price"})[["Open","High","Low","Price","Vol."]]
    df = df.astype(float)
    return df


def _fetch_coincap() -> pd.DataFrame:
    url = "https://api.coincap.io/v2/candles"
    params = {"exchange":"binance","interval":"m1","baseId":"bitcoin","quoteId":"tether","limit":TIME_STEP+50}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    records = resp.json().get("data", [])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["time"], unit='ms')
    df.set_index("Date", inplace=True)
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Price","volume":"Vol."})
    df = df[["Open","High","Low","Price","Vol."]].astype(float)
    return df


def _fetch_coingecko() -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency":"usd","days":2}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=["ts","Open","High","Low","Price"])
    df["Date"] = pd.to_datetime(df["ts"], unit='ms')
    df.set_index("Date", inplace=True)
    df = df.rename(columns={"Price":"Price"})
    df["Vol."] = 0.0
    df = df[["Open","High","Low","Price","Vol."]].astype(float)
    return df


def _load_live_data() -> pd.DataFrame:
    # 1) Binance REST
    try:
        df = _fetch_binance_rest()
        if len(df) >= TIME_STEP + 50:
            return df
    except:
        pass
    # 2) CoinCap
    try:
        df = _fetch_coincap()
        if len(df) >= TIME_STEP + 50:
            return df
    except:
        pass
    # 3) CoinGecko
    try:
        df = _fetch_coingecko()
        if len(df) >= TIME_STEP + 50:
            return df
    except:
        pass
    # 4) CSV fallback
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.rename(columns={"Price":"Price","Vol.":"Vol.","Open":"Open","High":"High","Low":"Low"})
    df = df[["Open","High","Low","Price","Vol."]].astype(float)
    return df


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.rsi(close=df["Price"], append=True)
    df.ta.macd(close=df["Price"], append=True)
    df["SMA_15"] = df["Price"].rolling(window=15).mean()
    return df.dropna()


def make_prediction() -> dict:
    df = _load_live_data().tail(TIME_STEP + 50)
    df = _prepare_dataframe(df)
    window = df.tail(TIME_STEP)
    if len(window) < TIME_STEP:
        return {"error": f"Need {TIME_STEP} rows; got {len(window)}"}

    features = list(_scaler.feature_names_in_)
    window = window[features]
    scaled = _scaler.transform(window)
    X = scaled.reshape(1, TIME_STEP, scaled.shape[1])
    pred_s = _model.predict(X, verbose=0)[0,0]
    dummy = np.zeros((1, scaled.shape[1])); dummy[0,0] = pred_s
    pred_p = _scaler.inverse_transform(dummy)[0,0]

    curr_p = window["Price"].iloc[-1]
    action, sl, tp = "HOLD", None, None
    buffer = BUY_THRESHOLD
    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action = "STRONG BUY"
        sl = curr_p * (1 - buffer)
        tp = pred_p * (1 - buffer)
    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action = "STRONG SELL"
        sl = curr_p * (1 + buffer)
        tp = pred_p * (1 + buffer)

    return {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": float(curr_p),
        "predicted_price": float(pred_p),
        "action": action,
        "stop_loss": sl,
        "take_profit": tp
    }
