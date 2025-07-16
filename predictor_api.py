# predictor_api.py

import os
import sys
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from joblib import load
from tensorflow.keras.models import load_model
import requests

# ─── PATH SETUP ─────────────────────────
BASE_DIR          = Path(__file__).parent
MODEL_DIR         = BASE_DIR / "Model1min"
CSV_PATH          = BASE_DIR / "data" / "BTC_DATA_V3.0.csv"
BINANCE_PROXY_URL = os.getenv("BINANCE_PROXY_URL")  # Cloudflare Worker URL

# ─── CONSTANTS ──────────────────────────
TIME_STEP      = 60
BUY_THRESHOLD  = 0.0002  # 0.02% threshold for signal
SELL_THRESHOLD = 0.0002

# ─── LOAD MODEL & SCALER ONCE ───────────
_model  = load_model(MODEL_DIR / "baru.h5")
_scaler = load(MODEL_DIR / "scaler.joblib")


def _fetch_binance_rest() -> pd.DataFrame:
    """
    Fetch latest 1m OHLCV via Worker proxy if set, otherwise direct.
    """
    base_url = BINANCE_PROXY_URL or "https://api.binance.com"
    url      = f"{base_url}/api/v3/klines"
    params   = {"symbol":"BTCUSDT","interval":"1m","limit":TIME_STEP+50}
    print(f"[DEBUG] _fetch_binance_rest using URL: {url}", file=sys.stderr)
    resp     = requests.get(url, params=params, timeout=10)
    print(f"[DEBUG] Binance REST GET {resp.url} -> {resp.status_code}", file=sys.stderr)
    resp.raise_for_status()
    data = resp.json()

    cols = [
        "open_time","Open","High","Low","Close","Vol.",
        "close_time","quote_asset_vol","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("Date", inplace=True)
    df = df.rename(columns={"Close":"Price"})[["Open","High","Low","Price","Vol."]]
    return df.astype(float)


def _fetch_coincap() -> pd.DataFrame:
    url = "https://api.coincap.io/v2/candles"
    params = {"exchange":"binance","interval":"m1","baseId":"bitcoin","quoteId":"tether","limit":TIME_STEP+50}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        print(f"[DEBUG] CoinCap GET -> {resp.status_code}", file=sys.stderr)
        rec = resp.json().get("data", [])
        df = pd.DataFrame(rec)
        df["Date"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("Date", inplace=True)
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Price","volume":"Vol."})
        return df[["Open","High","Low","Price","Vol."]].astype(float)
    except Exception as e:
        print(f"[DEBUG] _fetch_coincap failed: {e}", file=sys.stderr)
        raise


def _fetch_coingecko() -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency":"usd","days":2}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        print(f"[DEBUG] CoinGecko GET -> {resp.status_code}", file=sys.stderr)
        data = resp.json()
        df = pd.DataFrame(data, columns=["ts","Open","High","Low","Price"])
        df["Date"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("Date", inplace=True)
        df["Vol."] = 0.0
        return df[["Open","High","Low","Price","Vol."]].astype(float)
    except Exception as e:
        print(f"[DEBUG] _fetch_coingecko failed: {e}", file=sys.stderr)
        raise


def _load_live_data() -> pd.DataFrame:
    for fetch in (_fetch_binance_rest, _fetch_coincap, _fetch_coingecko):
        try:
            df = fetch()
            print(f"[DEBUG] {fetch.__name__} succeeded with {len(df)} rows", file=sys.stderr)
            if len(df) >= TIME_STEP + 50:
                return df
        except Exception:
            continue

    print("[DEBUG] Falling back to CSV", file=sys.stderr)
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.rename(columns={"Price":"Price","Vol.":"Vol.","Open":"Open","High":"High","Low":"Low"})
    return df[["Open","High","Low","Price","Vol."]].astype(float)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['Vol.'] = df['Vol.'].astype(float)
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
    window   = window[features]
    scaled   = _scaler.transform(window)
    X        = scaled.reshape(1, TIME_STEP, scaled.shape[1])
    pred_s   = _model.predict(X, verbose=0)[0,0]
    dummy    = np.zeros((1, scaled.shape[1])); dummy[0,0] = pred_s
    pred_p   = _scaler.inverse_transform(dummy)[0,0]

    curr_p = window["Price"].iloc[-1]
    action, sl, tp = "HOLD", None, None
    buffer = BUY_THRESHOLD
    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action = "STRONG BUY"
        sl     = curr_p * (1 - buffer)
        tp     = pred_p * (1 - buffer)
    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action = "STRONG SELL"
        sl     = curr_p * (1 + buffer)
        tp     = pred_p * (1 + buffer)

    return {
        "time":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":   float(curr_p),
        "predicted_price": float(pred_p),
        "action":          action,
        "stop_loss":       sl,
        "take_profit":     tp
    }
