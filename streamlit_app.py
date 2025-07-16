import streamlit as st
import pandas as pd
from predictor_api import make_prediction

# Page setup
st.set_page_config(page_title="BTC AI Predictor", layout="wide")
st.title("🔮 1-Hour BTC Price Prediction")

# Fetch
result = make_prediction()

if "error" in result:
    st.warning(result["error"])
else:
    # Layout
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Current Price",   f"${result['current_price']:,.2f}")
    col2.metric("Predicted Price", f"${result['predicted_price']:,.2f}")
    col3.metric("Suggestion",      result["action"])
    
    # Always compute SL/TP around current price
    sl = result["current_price"] * (1 - result.get("SL_PCT", 0.003))
    tp = result["current_price"] * (1 + result.get("TP_PCT", 0.009))
    
    # Build SL/TP table
    sltp_df = pd.DataFrame({
        "Type":  ["Stop Loss", "Take Profit"],
        "Value": [f"${sl:,.2f}", f"${tp:,.2f}"]
    })
    col4.table(sltp_df)

    # Last update
    st.markdown(f"*Last update:* {result['time']}")

st.markdown("---")

# Refresh button (this inherently reruns the script)
if st.button("🔄 Refresh Now"):
    pass
