import streamlit as st
import pandas as pd
import uvicorn
import threading
import time
from predictor_api import make_prediction
from api import app as fastapi_app

# --- Configuration ---
API_HOST = "0.0.0.0"
API_PORT = 8000

# --- Function to run the API ---
def run_api():
    """Starts the FastAPI server using uvicorn."""
    try:
        uvicorn.run(fastapi_app, host=API_HOST, port=API_PORT)
    except Exception as e:
        # This error will be printed in the console where streamlit is running
        print(f"Error starting API: {e}")

# --- Start API in a background thread if not already running ---
# We use st.session_state to ensure this only runs once per session.
if 'api_thread' not in st.session_state:
    print("Starting FastAPI server in a background thread.")
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    st.session_state.api_thread = api_thread
    # Give the server a moment to start up
    time.sleep(2)

# ─── Page setup ─────────────────────────
st.set_page_config(page_title="BTC AI Predictor", layout="wide")
st.title("🔮 1-Hour BTC Price Prediction")

# --- API Status in Sidebar ---
st.sidebar.header("API Status")
st.sidebar.success(f"✅ API Endpoint is running.")
st.sidebar.markdown(f"""
The prediction API is available for other applications to use.

**Endpoint URL:**
If you are running this on your local machine, the full URL is:
[`http://localhost:{API_PORT}/api.json`](http://localhost:{API_PORT}/api.json)
""")

# ─── Run prediction for Streamlit UI ───────────────────
result = make_prediction()

if "error" in result:
    st.warning(result["error"])
else:
    # ─ Layout ────────────────────────────
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Current Price",   f"${result['current_price']:,.2f}")
    col2.metric("Predicted Price", f"${result['predicted_price']:,.2f}")
    col3.metric("Suggestion",      result["action"])

    # ─ Build SL/TP table ─────────────────
    sl = result["stop_loss"]
    tp = result["take_profit"]

    sltp_df = pd.DataFrame({
        "Type":  ["Stop Loss",   "Take Profit"],
        "Value": [
            sl and f"${sl:,.2f}",  # will format or show blank if None
            tp and f"${tp:,.2f}"
        ]
    })

    col4.table(sltp_df)

    # ─ Last update timestamp ──────────────
    st.markdown(f"*Last update:* {result['time']}")

# ─ Divider & Refresh ────────────────────
st.markdown("---")
if st.button("🔄 Refresh Now"):
    st.rerun()