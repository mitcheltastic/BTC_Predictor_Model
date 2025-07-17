# streamlit_app.py
import streamlit as st
import pandas as pd
import requests  # <-- Make sure you have this installed (pip install requests)

# --- Configuration ---
# This is the address of your Flask API.
# If you deploy your Flask app, change this URL.
FLASK_API_URL = "http://127.0.0.1:5000/predict"

# --- Function to fetch data from your new API ---
def get_prediction_from_api():
    """
    Calls the Flask API to get the prediction data.
    Returns a dictionary with the prediction or an error message.
    """
    try:
        # Make a GET request to the /predict endpoint
        response = requests.get(FLASK_API_URL)
        # Raise an exception if the request returned an error (e.g., 404, 500)
        response.raise_for_status()
        # Return the JSON data from the response
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle connection errors or other request issues
        return {"error": f"Could not connect to the prediction API. Please ensure the Flask server is running. Details: {e}"}

# ─── Page setup ─────────────────────────
st.set_page_config(page_title="BTC AI Predictor", layout="wide")
st.title("🔮 1-Hour BTC Price Prediction")

# ─── Get prediction by calling the API ───────────────────
result = get_prediction_from_api()  # <-- This now calls your Flask API

if "error" in result:
    # Display any errors returned from the API call
    st.error(result["error"])
else:
    # ─ Layout (This part is mostly the same) ───────────
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Current Price",   f"${result.get('current_price', 0):,.2f}")
    col2.metric("Predicted Price", f"${result.get('predicted_price', 0):,.2f}")
    col3.metric("Suggestion",      result.get("action", "N/A"))

    # ─ Build SL/TP table ─────────────────
    # Use .get() to safely access keys that might be missing
    sl = result.get("stop_loss")
    tp = result.get("take_profit")

    sltp_df = pd.DataFrame({
        "Type":  ["Stop Loss", "Take Profit"],
        "Value": [
            f"${sl:,.2f}" if sl is not None else "N/A",
            f"${tp:,.2f}" if tp is not None else "N/A"
        ]
    })

    col4.table(sltp_df)

    # ─ Last update timestamp ──────────────
    st.markdown(f"*Last update:* {result.get('time', 'Unknown')}")

# ─ Divider & Refresh ────────────────────
st.markdown("---")
if st.button("🔄 Refresh Now"):
    st.rerun()

