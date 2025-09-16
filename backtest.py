import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from src.models.arima_model import ARIMAModel
from src.evaluation.metrics import calculate_metrics
import warnings
warnings.filterwarnings('ignore')

def backtest_model(symbol, test_periods=10, days_ahead=7):
    """
    Backtest the model over multiple periods to get realistic performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Backtesting {symbol} - {test_periods} periods, {days_ahead} days ahead")
    print('='*60)
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)['Close']
    
    all_metrics = []
    predictions_vs_actuals = []
    
    # Walk-forward validation
    for i in range(test_periods):
        test_end = len(data) - (i * days_ahead)
        test_start = test_end - days_ahead
        train_end = test_start
        train_start = max(0, train_end - 60)  # Use 60 days for training
        
        if train_start < 0 or test_start < 0 or test_end > len(data):
            break
            
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        
        if len(train_data) < 10 or len(test_data) == 0:
            break
        
        try:
            # Train model
            model = ARIMAModel()
            model.find_best_params(train_data, 
                                  p_range=range(0, 3),
                                  d_range=range(0, 2), 
                                  q_range=range(0, 3))
            model.train(train_data)
            
            # Predict
            predictions = model.predict(len(test_data))
            
            # Ensure predictions is a numpy array
            if isinstance(predictions, pd.Series):
                predictions = predictions.values
            
            # Calculate metrics
            metrics = calculate_metrics(test_data.values, predictions)
            all_metrics.append(metrics)
            
            # Store predictions vs actuals
            for j in range(min(len(test_data), len(predictions))):
                predictions_vs_actuals.append({
                    'actual': test_data.iloc[j],
                    'predicted': predictions[j],  # Now using numpy array indexing
                    'error': abs(test_data.iloc[j] - predictions[j]),
                    'percent_error': abs((test_data.iloc[j] - predictions[j]) / test_data.iloc[j] * 100)
                })
            
            print(f"Period {i+1}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%")
            
        except Exception as e:
            print(f"Period {i+1}: Error - {str(e)}")
            continue
    
    if len(all_metrics) == 0:
        print("No successful predictions made")
        return None, []
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print(f"\n{'='*60}")
    print("AVERAGE PERFORMANCE ACROSS ALL PERIODS:")
    print(f"RMSE: {avg_metrics['RMSE']:.2f}")
    print(f"MAE: {avg_metrics['MAE']:.2f}")
    print(f"MAPE: {avg_metrics['MAPE']:.2f}%")
    print(f"Directional Accuracy: {avg_metrics['Directional_Accuracy']:.1f}%")
    print('='*60)
    
    # Analysis of errors
    if predictions_vs_actuals:
        errors = [p['percent_error'] for p in predictions_vs_actuals]
        print(f"\nError Distribution:")
        print(f"Min Error: {min(errors):.2f}%")
        print(f"Max Error: {max(errors):.2f}%")
        print(f"Median Error: {np.median(errors):.2f}%")
        print(f"Errors < 2%: {sum(1 for e in errors if e < 2)} ({sum(1 for e in errors if e < 2)/len(errors)*100:.1f}%)")
        print(f"Errors < 5%: {sum(1 for e in errors if e < 5)} ({sum(1 for e in errors if e < 5)/len(errors)*100:.1f}%)")
    
    return avg_metrics, predictions_vs_actuals

# Test multiple symbols
if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    results = {}
    
    for symbol in symbols:
        result = backtest_model(symbol, test_periods=5, days_ahead=7)
        if result[0] is not None:
            results[symbol] = result[0]
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Average 7-day prediction accuracy:")
    print("="*60)
    for symbol, metrics in results.items():
        print(f"{symbol:5} -> MAPE: {metrics['MAPE']:5.2f}% | Direction: {metrics['Directional_Accuracy']:5.1f}% | RMSE: ${metrics['RMSE']:6.2f}")
    
    # Identify best performing stock
    if results:
        best_stock = min(results.items(), key=lambda x: x[1]['MAPE'])
        print(f"\nBest performing: {best_stock[0]} with MAPE of {best_stock[1]['MAPE']:.2f}%")
