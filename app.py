import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from src with full path
from src.data.data_processor import fetch_stock_data
from src.models.arima_model import ARIMAModel
from src.evaluation.metrics import calculate_metrics
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

st.title("📈 Stock Price Prediction Dashboard")
st.markdown("### AI-Powered Stock Price Forecasting using ARIMA Models")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    symbol = st.selectbox(
        "Select Stock Symbol",
        ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"],
        help="Choose the stock symbol to analyze"
    )
    
    days_back = st.slider(
        "Historical Days",
        min_value=30,
        max_value=365,
        value=60,
        help="Number of days of historical data to use"
    )
    
    prediction_days = st.slider(
        "Prediction Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to predict into the future"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This app uses ARIMA models to predict stock prices. "
        "ARIMA captures time series patterns and trends."
    )

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Selected Stock", symbol)
with col2:
    st.metric("Historical Days", days_back)
with col3:
    st.metric("Prediction Days", prediction_days)

if st.button("🚀 Run Prediction", type="primary"):
    with st.spinner(f"Training ARIMA model for {symbol}..."):
        try:
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back * 2)
            
            data = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            data = data.tail(days_back)
            
            # Split data
            split_point = int(len(data) * 0.9)
            train_data = data['Close'][:split_point]
            test_data = data['Close'][split_point:]
            
            # Train ARIMA model
            arima_model = ARIMAModel()
            best_order = arima_model.find_best_params(train_data, 
                                                      p_range=range(0, 5),
                                                      d_range=range(0, 2), 
                                                      q_range=range(0, 5))
            arima_model.train(train_data)
            
            # Make predictions
            test_predictions = arima_model.predict(len(test_data))
            future_predictions = arima_model.predict(prediction_days)
            
            # Calculate metrics
            metrics = calculate_metrics(test_data.values, test_predictions)
            
            st.success("✅ Prediction completed successfully!")
            
            # Display metrics
            st.markdown("### 📊 Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col2.metric("MAE", f"{metrics['MAE']:.2f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            col4.metric("Direction Accuracy", f"{metrics['Directional_Accuracy']:.1f}%")
            
            # Create interactive plot
            st.markdown("### 📈 Stock Price Predictions")
            
            fig = go.Figure()
            
            # Add historical prices
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'].values,
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Add test data
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data.values,
                mode='lines',
                name='Actual (Test)',
                line=dict(color='green', width=2)
            ))
            
            # Add test predictions
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_predictions,
                mode='lines',
                name='ARIMA Predicted',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add future predictions
            future_dates = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), 
                                        periods=prediction_days)
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='Future Predictions',
                line=dict(color='orange', width=2, dash='dot'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price Predictions - ARIMA{best_order}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction table
            st.markdown("### 🔮 Future Price Predictions")
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_predictions
            })
            future_df['Predicted Price'] = future_df['Predicted Price'].round(2)
            st.dataframe(future_df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, PyTorch, and Statsmodels")
