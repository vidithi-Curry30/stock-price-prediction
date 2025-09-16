import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance"""
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    # Set frequency to business days to avoid warnings
    data = data.asfreq('B', method='ffill')
    return data

def prepare_data(data: pd.DataFrame, feature_col: str = 'Close') -> Tuple[pd.Series, pd.Series]:
    """Prepare data for model training"""
    # Use closing price as the feature
    prices = data[feature_col].dropna()
    
    # Split into train and test (80/20)
    split_point = int(len(prices) * 0.8)
    train_data = prices[:split_point]
    test_data = prices[split_point:]
    
    return train_data, test_data
