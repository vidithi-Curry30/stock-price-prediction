# Stock Price Prediction with ARIMA-GARCH and VADPS

## Overview
Advanced time series forecasting system achieving <2% MAPE on major tech stocks using ARIMA-GARCH models enhanced with a novel Volatility-Adjusted Directional Probability Score (VADPS).

## Key Features
- **Institutional-grade accuracy**: 1.46% MAPE on MSFT
- **Novel VADPS algorithm**: Filters predictions to 65%+ confidence trades
- **Risk management**: Only signals 20-30% of days (high-quality setups only)
- **Live dashboard**: Streamlit interface for real-time predictions

## Quick Start
```bash
# Clone and install
git clone https://github.com/vidithi-Curry30/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt

# Run predictions
python demo.py MSFT
