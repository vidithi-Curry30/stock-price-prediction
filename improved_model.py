import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.models.arima_model import ARIMAModel
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class ImprovedStockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.models = []
        
    def add_features(self, data):
        """Add technical indicators"""
        df = pd.DataFrame(data)
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(5).std()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        return df
    
    def detect_trend(self, data):
        """Detect if stock is trending or ranging"""
        recent = data[-20:]
        trend_strength = abs(recent[-1] - recent[0]) / recent.std()
        return 'trending' if trend_strength > 1.5 else 'ranging'
    
    def adaptive_predict(self, train_data, test_days=7):
        """Use different strategies based on market conditions"""
        
        # Detect market regime
        regime = self.detect_trend(train_data)
        print(f"Market regime detected: {regime}")
        
        predictions = []
        
        # Model 1: Standard ARIMA
        model1 = ARIMAModel()
        if regime == 'trending':
            # Use more aggressive parameters for trending
            model1.order = (2, 1, 2)
        else:
            # Conservative for ranging markets
            model1.order = (1, 1, 1)
        
        model1.train(train_data)
        pred1 = model1.predict(test_days)
        predictions.append(pred1)
        
        # Model 2: Short-term ARIMA (last 30 days only)
        model2 = ARIMAModel()
        model2.train(train_data[-30:])
        pred2 = model2.predict(test_days)
        predictions.append(pred2)
        
        # Model 3: Long-term ARIMA (if enough data)
        if len(train_data) > 100:
            model3 = ARIMAModel()
            model3.train(train_data[-100:])
            pred3 = model3.predict(test_days)
            predictions.append(pred3)
        
        # Weighted average based on regime
        if regime == 'trending':
            # Give more weight to longer-term models
            weights = [0.3, 0.3, 0.4] if len(predictions) == 3 else [0.4, 0.6]
        else:
            # Give more weight to short-term model in ranging market
            weights = [0.3, 0.5, 0.2] if len(predictions) == 3 else [0.3, 0.7]
        
        # Ensure predictions are numpy arrays
        predictions = [p.values if hasattr(p, 'values') else p for p in predictions]
        
        # Calculate weighted average
        final_predictions = np.average(predictions, axis=0, weights=weights[:len(predictions)])
        
        return final_predictions, regime

def test_improved_model():
    """Test the improved model"""
    print("\n" + "="*60)
    print("TESTING IMPROVED ADAPTIVE MODEL")
    print("="*60)
    
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        print(f"\n{symbol}:")
        print("-"*40)
        
        # Get data
        end = datetime.now()
        start = end - timedelta(days=180)
        stock = yf.Ticker(symbol)
        data = stock.history(start=start, end=end)['Close']
        
        # Split
        train = data[:-7]
        test = data[-7:]
        
        # Predict with improved model
        model = ImprovedStockPredictor(symbol)
        predictions, regime = model.adaptive_predict(train, 7)
        
        # Calculate errors
        errors = []
        for i in range(min(len(test), len(predictions))):
            actual = test.iloc[i]
            pred = predictions[i]
            error = abs(actual - pred) / actual * 100
            errors.append(error)
            
        print(f"Average Error: {np.mean(errors):.2f}%")
        print(f"Max Error: {max(errors):.2f}%")
        print(f"Errors < 3%: {sum(1 for e in errors if e < 3)}/7")

if __name__ == "__main__":
    test_improved_model()
