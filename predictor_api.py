import os
import datetime
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from joblib import load
from tensorflow.keras.models import load_model
import requests
import pytz

# --- PATH SETUP ---
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "Model1min"
CSV_PATH  = BASE_DIR / "data" / "BTC_DATA_V3.0.csv"

# --- CONSTANTS ---
TIME_STEP      = 60
BUY_THRESHOLD  = 0.0002  # 0.02% threshold for signal
SELL_THRESHOLD = 0.0002

# --- LOAD MODEL & SCALER ONCE ---
_model  = load_model(MODEL_DIR / "baru.h5")
_scaler = load(MODEL_DIR / "scaler.joblib")


def _fetch_kraken() -> pd.DataFrame:
    """
    Fetch latest 1m OHLCV from Kraken Public API (unblocked).
    """
    import time
    since = int(time.time()) - (TIME_STEP + 50) * 60
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSDT", "interval": 1, "since": since}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # extract OHLC array for XBTUSDT
    ohlc = list(data.get('result', {}).values())[0]
    df = pd.DataFrame(ohlc, columns=["time","Open","High","Low","Close","Vwap","Volume","Count"])
    
    print(f"Successfully fetched {len(df)} rows of data from Kraken.")

    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('Date', inplace=True)
    df = df.rename(columns={'Close':'Price','Volume':'Vol.'})[["Open","High","Low","Price","Vol."]]
    return df.astype(float)


def _load_live_data() -> pd.DataFrame:
    
    """
    Try Kraken first, then fallback to local CSV.
    """
    try:
        df = _fetch_kraken()
        if len(df) >= TIME_STEP + 50:
            return df
    except Exception as e:
        print(f"Could not fetch from Kraken, falling back to CSV. Error: {e}")
        pass

    # Fallback to static CSV
    print("Fetching data from local CSV file.")
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.rename(columns={
        'Price':'Price', 'Vol.':'Vol.', 'Open':'Open', 'High':'High', 'Low':'Low'
    })[["Open","High","Low","Price","Vol."]]
    return df.astype(float)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and clean data.
    """
    df['Vol.'] = df['Vol.'].astype(float)
    df.ta.rsi(close=df['Price'], append=True)
    df.ta.macd(close=df['Price'], append=True)
    df['SMA_15'] = df['Price'].rolling(window=15).mean()
    return df.dropna()


def make_prediction() -> dict:
    """
    Fetch live data, prepare inputs, run the model,
    and return a summary dict with time, prices, action, SL, TP.
    """
    # Load and trim data
    df = _load_live_data().tail(TIME_STEP + 50)

    # Compute indicators
    df = _prepare_dataframe(df)

    # Build prediction window
    window = df.tail(TIME_STEP)
    if len(window) < TIME_STEP:
        return {"error": f"Need {TIME_STEP} rows; got {len(window)}"}

    # Scale features
    feature_cols = list(_scaler.feature_names_in_)
    data_in = window[feature_cols]
    scaled  = _scaler.transform(data_in)

    # Predict
    X = scaled.reshape(1, TIME_STEP, scaled.shape[1])
    pred_s = _model.predict(X, verbose=0)[0,0]

    # Inverse scale
    dummy  = np.zeros((1, scaled.shape[1])); dummy[0,0] = pred_s
    pred_p = _scaler.inverse_transform(dummy)[0,0]

    # Current price
    curr_p = window['Price'].iloc[-1]

    # Trading logic with tight SL/TP
    action, sl, tp = "HOLD", None, None
    buffer = BUY_THRESHOLD

    if pred_p > curr_p * (1 + BUY_THRESHOLD):
        action = "BUY"
        sl     = curr_p * (1 - buffer)
        tp     = pred_p * (1 - buffer)
    elif pred_p < curr_p * (1 - SELL_THRESHOLD):
        action = "SELL"
        sl     = curr_p * (1 + buffer)
        tp     = pred_p * (1 + buffer)

    # --- TIMEZONE ADJUSTMENT ---
    jakarta_tz = pytz.timezone('Asia/Jakarta')
    now_jakarta = datetime.datetime.now(jakarta_tz)

    return {
        "time":              now_jakarta.strftime("%Y-%m-%d %H:%M:%S"),
        "current_price":   float(curr_p),
        "predicted_price": float(pred_p),
        "action":            action,
        "stop_loss":       sl,
        "take_profit":     tp
    }
