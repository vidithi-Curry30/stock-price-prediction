# Stock Price Prediction

A machine learning project for predicting stock prices using ARIMA and GARCH models.

## What This Does

This project predicts stock prices 1-7 days into the future. After testing multiple approaches, I found that combining ARIMA (for price trends) with GARCH (for volatility) gives the best results.

Current accuracy on recent tests:
- MSFT: 1.46% average error
- AAPL: 1.79% average error  
- GOOGL: 2.75% average error

## How to Use It

1. Clone the repo and install dependencies:
```bash
git clone https://github.com/vidithi-Curry30/stock-price-prediction.git
cd stock-price-prediction
pip install -r requirements.txt
