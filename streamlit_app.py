import streamlit as st
from predictor_api import make_prediction
import pandas as pd

# Page setup
st.set_page_config(page_title="BTC AI Predictor", layout="wide")
st.title("🔮 1-Hour BTC Price Prediction")

# Fetch data
result = make_prediction()

if "error" in result:
    st.warning(result["error"])
else:
    c1, c2, c3, c4 = st.columns(4, gap="large")
    c1.metric("Current Price",   f"${result['current_price']:,.2f}")
    c2.metric("Predicted Price", f"${result['predicted_price']:,.2f}")
    c3.metric("Suggestion",      result["action"])
    
    sl = result.get("stop_loss")
    tp = result.get("take_profit")

    if sl is not None and tp is not None:
        # Build a small DataFrame for SL/TP
        df_sl_tp = pd.DataFrame({
            "Type":  ["Stop Loss", "Take Profit"],
            "Value": [f"${sl:,.2f}",  f"${tp:,.2f}"]
        })
        c4.table(df_sl_tp)
    else:
        c4.write("—")

    st.markdown(f"*Last update:* {result['time']}")

st.markdown("---")
if st.button("🔄 Refresh Now"):
    st.experimental_rerun()
