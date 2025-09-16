import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.models.arima_garch_model import ARIMAGARCHModel
from src.evaluation.metrics import calculate_metrics

def comprehensive_test(symbol='MSFT', test_periods=5):
    """Run comprehensive tests on ARIMA+GARCH model"""
    
    print(f"\n{'='*60}")
    print(f"TESTING ARIMA+GARCH MODEL FOR {symbol}")
    print('='*60)
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)['Close']
    
    all_results = []
    
    # Backtesting loop
    for period in range(test_periods):
        print(f"\n--- Test Period {period + 1} ---")
        
        # Create train/test split
        test_size = 7
        test_end = len(data) - (period * test_size)
        test_start = test_end - test_size
        train_data = data[:test_start]
        test_data = data[test_start:test_end]
        
        if len(train_data) < 60:
            break
        
        # Train model
        model = ARIMAGARCHModel()
        residual_analysis = model.train(train_data)
        
        # Make predictions
        predictions = model.predict(steps=len(test_data))
        
        # Calculate metrics
        metrics = calculate_metrics(test_data.values, predictions['mean'])
        
        # Check if predictions fall within confidence intervals
        within_95 = 0
        within_99 = 0
        for i in range(len(test_data)):
            actual = test_data.iloc[i]
            if predictions['lower_95'][i] <= actual <= predictions['upper_95'][i]:
                within_95 += 1
            if predictions['lower_99'][i] <= actual <= predictions['upper_99'][i]:
                within_99 += 1
        
        metrics['within_95_ci'] = within_95 / len(test_data) * 100
        metrics['within_99_ci'] = within_99 / len(test_data) * 100
        
        all_results.append(metrics)
        
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Within 95% CI: {metrics['within_95_ci']:.1f}%")
        print(f"Within 99% CI: {metrics['within_99_ci']:.1f}%")
    
    # Average results
    avg_metrics = {}
    for key in all_results[0].keys():
        avg_metrics[key] = np.mean([r[key] for r in all_results])
    
    print(f"\n{'='*60}")
    print("AVERAGE PERFORMANCE:")
    print(f"MAPE: {avg_metrics['MAPE']:.2f}%")
    print(f"RMSE: {avg_metrics['RMSE']:.2f}")
    print(f"Direction Accuracy: {avg_metrics['Directional_Accuracy']:.1f}%")
    print(f"95% CI Coverage: {avg_metrics['within_95_ci']:.1f}%")
    print(f"99% CI Coverage: {avg_metrics['within_99_ci']:.1f}%")
    print('='*60)
    
    return avg_metrics

def plot_predictions_with_ci(symbol='MSFT'):
    """Create visualization with confidence intervals"""
    
    # Get recent data
    stock = yf.Ticker(symbol)
    data = stock.history(period="3mo")['Close']
    
    # Train model
    train_data = data[:-7]
    test_data = data[-7:]
    
    model = ARIMAGARCHModel()
    model.train(train_data)
    predictions = model.predict(7)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Predictions with confidence intervals
    ax1.plot(train_data.index[-30:], train_data.values[-30:], 'b-', label='Historical')
    ax1.plot(test_data.index, test_data.values, 'g-', label='Actual', linewidth=2)
    
    future_dates = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1), periods=7)
    ax1.plot(future_dates, predictions['mean'], 'r--', label='Predicted', linewidth=2)
    
    # Add confidence intervals
    ax1.fill_between(future_dates, predictions['lower_95'], predictions['upper_95'], 
                     alpha=0.3, color='red', label='95% CI')
    ax1.fill_between(future_dates, predictions['lower_99'], predictions['upper_99'], 
                     alpha=0.2, color='red', label='99% CI')
    
    ax1.set_title(f'{symbol} ARIMA+GARCH Predictions with Confidence Intervals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility forecast
    ax2.bar(future_dates, predictions['volatility'], color='orange', alpha=0.6)
    ax2.set_title('Predicted Volatility (GARCH)')
    ax2.set_ylabel('Conditional Volatility')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arima_garch_predictions.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n7-Day Predictions for {symbol}:")
    for i in range(len(predictions['mean'])):
        print(f"Day {i+1}: ${predictions['mean'][i]:.2f} "
              f"[95% CI: ${predictions['lower_95'][i]:.2f}-${predictions['upper_95'][i]:.2f}]")

# Run tests
if __name__ == "__main__":
    # Test on multiple stocks
    for symbol in ['MSFT', 'AAPL', 'GOOGL']:
        comprehensive_test(symbol, test_periods=3)
    
    # Create visualization for MSFT
    plot_predictions_with_ci('MSFT')
