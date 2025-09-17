import yfinance as yf
from datetime import datetime, timedelta
from src.models.arima_model import ARIMAModel
import numpy as np

def quick_accuracy_test(symbol='AAPL'):
    """Test on the most recent data"""
    print(f"\nQuick test for {symbol}")
    print("="*40)
    
    # Get recent data
    end = datetime.now()
    start = end - timedelta(days=90)
    
    stock = yf.Ticker(symbol)
    data = stock.history(start=start, end=end)['Close']
    
    # Use last 7 days as test
    train = data[:-7]
    test = data[-7:]
    
    # Train and predict
    model = ARIMAModel()
    model.train(train)
    predictions = model.predict(7)
    
    # Convert to numpy array if needed
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    
    # Show results
    for i in range(min(len(test), len(predictions))):
        actual = test.iloc[i]
        pred = predictions[i]
        error = abs(actual - pred) / actual * 100
        print(f"Day {i+1}: Actual=${actual:.2f}, Predicted=${pred:.2f}, Error={error:.2f}%")
    
    errors = [abs(test.iloc[i] - predictions[i])/test.iloc[i]*100 for i in range(min(len(test), len(predictions)))]
    print(f"\nAverage Error: {np.mean(errors):.2f}%")
    print(f"Best day: {min(errors):.2f}% error")
    print(f"Worst day: {max(errors):.2f}% error")

# Run quick test
for symbol in ['AAPL', 'MSFT']:
    quick_accuracy_test(symbol)
