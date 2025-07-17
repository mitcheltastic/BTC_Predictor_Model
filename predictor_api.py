# predictor_api.py
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime
import pytz

# --- Constants ---
MODEL_PATH = "Model1min/baru.h5"
SCALER_PATH = "Model1min/scaler.joblib"
TIMEZONE = pytz.timezone("Asia/Jakarta") # Adjust to your timezone if needed

def load_heavy_assets():
    """
    Loads the pre-trained model and scaler from disk.
    This is the slow, memory-intensive part that we will now do only ONCE.
    """
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def make_prediction(model, scaler):
    """
    Generates a prediction using the pre-loaded model and scaler.
    This function is now very fast.
    """
    try:
        # 1. Fetch live data
        btc_data = yf.download(tickers='BTC-USD', period='2d', interval='1h')
        if btc_data.empty:
            return {"error": "Failed to fetch live BTC data."}

        # 2. Add technical indicators
        btc_data.ta.rsi(append=True)
        btc_data.ta.macd(append=True)
        btc_data.ta.bbands(append=True)
        btc_data.dropna(inplace=True)

        # 3. Get the most recent data point for prediction
        last_row = btc_data.iloc[[-1]]
        current_price = last_row['Close'].iloc[0]

        # 4. Scale the features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']
        features = last_row[feature_columns]
        scaled_features = scaler.transform(features)
        scaled_features = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))

        # 5. Make prediction
        prediction_scaled = model.predict(scaled_features)
        
        # Create a dummy array to inverse transform the prediction
        dummy_array = np.zeros((1, len(feature_columns)))
        dummy_array[0, 3] = prediction_scaled[0, 0] # Index 3 is 'Close'
        predicted_price = scaler.inverse_transform(dummy_array)[0, 3]

        # 6. Determine action and SL/TP
        action = "BUY" if predicted_price > current_price else "SELL"
        stop_loss = current_price * 0.998 if action == "BUY" else current_price * 1.002
        take_profit = current_price * 1.002 if action == "BUY" else current_price * 0.998

        return {
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "action": action,
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "time": datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}

