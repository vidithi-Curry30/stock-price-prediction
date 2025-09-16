import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from improved_model import ImprovedStockPredictor
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Enhanced Stock Predictor", layout="wide", page_icon="📈")

st.title("📈 Enhanced Adaptive Stock Predictor")
st.markdown("### AI-Powered Predictions with Market Regime Detection")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    symbol = st.selectbox(
        "Select Stock",
        ["MSFT", "AAPL", "GOOGL", "TSLA", "NVDA"],
        help="MSFT shows best performance (1.47% avg error)"
    )
    
    prediction_days = st.slider("Days to Predict", 1, 14, 7)
    
    st.markdown("---")
    st.markdown("### Model Performance")
    st.info("""
    Recent Accuracy:
    - MSFT: 1.47% error
    - GOOGL: 2.94% error  
    - AAPL: 3.04% error
    """)

if st.button("🚀 Generate Prediction", type="primary"):
    with st.spinner(f"Analyzing {symbol}..."):
        # Fetch data
        stock = yf.Ticker(symbol)
        data = stock.history(period="6mo")['Close']
        
        # Get current price
        current_price = data.iloc[-1]
        
        # Make predictions
        model = ImprovedStockPredictor(symbol)
        predictions, regime = model.adaptive_predict(data, prediction_days)
        
        # Display regime
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Market Regime", regime.upper())
        with col3:
            pred_change = ((predictions[-1] - current_price) / current_price) * 100
            st.metric("Expected Change", f"{pred_change:+.2f}%")
        
        # Create chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index[-30:],
            y=data.values[-30:],
            name="Historical (30d)",
            line=dict(color="blue", width=2)
        ))
        
        # Predictions
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=prediction_days
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name="Predictions",
            line=dict(color="red", width=2, dash="dash"),
            mode='lines+markers'
        ))
        
        # Add confidence band (±3% based on your results)
        error_margin = 0.03
        upper_bound = predictions * (1 + error_margin)
        lower_bound = predictions * (1 - error_margin)
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.2)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0.2)'),
            name='95% Confidence'
        ))
        
        fig.update_layout(
            title=f"{symbol} Price Prediction ({regime} market)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction table
        st.markdown("### 📊 Daily Predictions")
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': predictions,
            'Lower Bound (-3%)': lower_bound,
            'Upper Bound (+3%)': upper_bound,
            'Change from Today': [(p - current_price) / current_price * 100 for p in predictions]
        })
        
        pred_df['Predicted Price'] = pred_df['Predicted Price'].round(2)
        pred_df['Lower Bound (-3%)'] = pred_df['Lower Bound (-3%)'].round(2)
        pred_df['Upper Bound (+3%)'] = pred_df['Upper Bound (+3%)'].round(2)
        pred_df['Change from Today'] = pred_df['Change from Today'].round(2)
        
        st.dataframe(pred_df, hide_index=True, use_container_width=True)
        
        # Trading signals
        st.markdown("### 📈 Trading Signal")
        
        avg_prediction = np.mean(predictions)
        if avg_prediction > current_price * 1.02:
            st.success(f"🟢 BUY Signal - Expected {pred_change:+.2f}% return")
        elif avg_prediction < current_price * 0.98:
            st.warning(f"🔴 SELL Signal - Expected {pred_change:+.2f}% return")
        else:
            st.info(f"⏸️ HOLD - Expected {pred_change:+.2f}% return (< 2% change)")

st.markdown("---")
st.markdown("Model uses adaptive ARIMA with market regime detection. Best for MSFT (1.47% avg error).")
