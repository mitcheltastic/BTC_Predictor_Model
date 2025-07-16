# app.py

import streamlit as st
from predictor_api import make_prediction

# Page setup
st.set_page_config(page_title="BTC AI Predictor", layout="wide")
st.title("🔮 1-Hour BTC Price Prediction")

# Fetch data
result = make_prediction()

if "error" in result:
    st.warning(result["error"])
else:
    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Current Price",   f"${result['current_price']:,.2f}")
    col2.metric("Predicted Price", f"${result['predicted_price']:,.2f}")
    col3.metric("Suggestion",      result["action"])
    
    # Display SL & TP as a table
    sl, tp = result["stop_loss"], result["take_profit"]
    sl_tp_df = {
        "Type": ["Stop Loss", "Take Profit"],
        "Value": [f"${sl:,.2f}", f"${tp:,.2f}"]
    }
    col4.table(sl_tp_df)
    
    st.markdown(f"*Last update:* {result['time']}")

st.markdown("---")
# Manual refresh button (no need to call st.experimental_rerun)
if st.button("🔄 Refresh Now"):
    pass  # clicking this widget will automatically rerun the script
