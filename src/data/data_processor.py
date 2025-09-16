import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

class StockDataProcessor:
    """Handles data collection, preprocessing, and feature engineering"""
    
    def __init__(self, symbol='AAPL', period='5y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        print(f"Fetched {len(self.data)} days of data for {self.symbol}")
        return self.data
    
    def add_technical_indicators(self):
        """Add technical indicators as features"""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
            
        # Moving averages
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        
        # Volatility
        self.data['Volatility'] = self.data['Close'].rolling(window=10).std()
        
        # RSI
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        
        # Price change
        self.data['Price_Change'] = self.data['Close'].pct_change()
        
        # Volume indicators
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=10).mean()
        
        return self.data
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def check_stationarity(self, series):
        """Check if time series is stationary using ADF test"""
        result = adfuller(series.dropna())
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        
        if result[1] <= 0.05:
            print("Series is stationary")
            return True
        else:
            print("Series is not stationary")
            return False
