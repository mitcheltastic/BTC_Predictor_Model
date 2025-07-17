import streamlit as st
import pandas as pd
from predictor_api import make_prediction

# --- API Endpoint Logic ---
# Check if the 'endpoint' query parameter is set to 'api'
if st.query_params.get("endpoint") == "api":
    # Run the prediction
    result = make_prediction()
    # Return the result as JSON and stop the script
    st.json(result)
    st.stop()

# --- Regular Streamlit UI (runs if the endpoint parameter is not present) ---

# ─── Page setup ─────────────────────────
st.set_page_config(page_title="BTC AI Predictor", layout="wide")
st.title("🔮 1-Hour BTC Price Prediction")

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

