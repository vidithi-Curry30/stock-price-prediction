import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

class StockDataProcessor:
    def __init__(self, symbol='AAPL', period='5y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        print(f"Fetched {len(self.data)} days of data for {self.symbol}")
        return self.data
    
    def add_technical_indicators(self):
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
            
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['Volatility'] = self.data['Close'].rolling(window=10).std()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=10).mean()
        return self.data
    
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
