import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import warnings
warnings.filterwarnings('ignore')

from data.data_processor import fetch_stock_data, prepare_data
from models.arima_model import ARIMAModel
from models.lstm_model import StockLSTM
from evaluation.metrics import calculate_metrics
from datetime import datetime, timedelta

def run_analysis(symbol='AAPL', days_back=90, prediction_days=7, model_type='both'):
    """
    Run stock price prediction analysis
    
    Args:
        symbol: Stock symbol
        days_back: Number of days of historical data to use
        prediction_days: Number of days to predict forward
        model_type: 'arima', 'lstm', or 'both'
    """
    
    # Set up MLflow
    mlflow.set_experiment("stock_price_prediction")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("days_back", days_back)
        mlflow.log_param("prediction_days", prediction_days)
        
        print(f"Starting analysis for {symbol}")
        
        # 1. Fetch recent data (shorter timeframe for better ARIMA performance)
        print(f"1. Fetching last {days_back} days of data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back * 2)  # Get extra to ensure enough trading days
        
        data = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        data = data.tail(days_back)  # Use only last N days
        print(f"Fetched {len(data)} days of data for {symbol}")
        
        # 2. Split data for validation (90% train, 10% test for short-term)
        split_point = int(len(data) * 0.9)
        train_data = data['Close'][:split_point]
        test_data = data['Close'][split_point:]
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        results = {}
        
        # 3. Train ARIMA model (better for short-term)
        if model_type in ['arima', 'both']:
            print("\n2. Training ARIMA model...")
            arima_model = ARIMAModel()
            
            # Use smaller parameter ranges for short-term data
            arima_model.find_best_params(train_data, 
                                        p_range=range(0, 5),
                                        d_range=range(0, 2), 
                                        q_range=range(0, 5))
            arima_model.train(train_data)
            
            # Make predictions
            arima_predictions = arima_model.predict(len(test_data))
            
            # Calculate metrics
            arima_metrics = calculate_metrics(test_data.values, arima_predictions)
            print(f"ARIMA Metrics: {arima_metrics}")
            
            # Log metrics
            for metric_name, value in arima_metrics.items():
                mlflow.log_metric(f"arima_{metric_name}", value)
            
            # Make future predictions
            future_predictions = arima_model.predict(prediction_days)
            
            results['arima'] = {
                'test_predictions': arima_predictions,
                'future_predictions': future_predictions,
                'metrics': arima_metrics,
                'test_data': test_data
            }
        
        # 4. Train LSTM model
        if model_type in ['lstm', 'both']:
            print("\n3. Training LSTM model...")
            
            # Use smaller sequence length for short-term data
            sequence_length = min(20, len(train_data) // 3)
            
            lstm_model = StockLSTM(
                sequence_length=sequence_length,
                hidden_dim=32,  # Smaller network for less data
                num_layers=2,
                epochs=100,
                learning_rate=0.01
            )
            
            # Train and predict
            lstm_model.train(train_data)
            lstm_predictions = lstm_model.predict(test_data, train_data)
            
            # Calculate metrics
            lstm_metrics = calculate_metrics(test_data.values, lstm_predictions[:len(test_data)])
            print(f"LSTM Metrics: {lstm_metrics}")
            
            # Log metrics
            for metric_name, value in lstm_metrics.items():
                mlflow.log_metric(f"lstm_{metric_name}", value)
            
            results['lstm'] = {
                'test_predictions': lstm_predictions[:len(test_data)],
                'metrics': lstm_metrics,
                'test_data': test_data
            }
        
        # 5. Create better visualization
        print("\n4. Creating visualizations...")
        create_visualization(data, results, symbol, prediction_days)
        
        return results

def create_visualization(full_data, results, symbol, prediction_days):
    """Create improved visualization with actual vs predicted"""
    
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 7*len(results)))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot full historical data
        ax.plot(full_data.index, full_data['Close'].values, 
               label='Historical', color='blue', alpha=0.7, linewidth=1.5)
        
        # Plot test data
        test_data = result['test_data']
        ax.plot(test_data.index, test_data.values, 
               label='Actual (Test)', color='green', linewidth=2)
        
        # Plot predictions on test data
        ax.plot(test_data.index, result['test_predictions'], 
               label=f'{model_name.upper()} Predicted', color='red', 
               linewidth=2, linestyle='--')
        
        # Add future predictions if available
        if 'future_predictions' in result:
            future_dates = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), 
                                        periods=prediction_days)
            ax.plot(future_dates, result['future_predictions'], 
                   label='Future Predictions', color='orange', 
                   linewidth=2, linestyle=':')
            ax.axvspan(future_dates[0], future_dates[-1], alpha=0.1, color='orange')
        
        # Formatting
        ax.set_title(f'{symbol} {model_name.upper()} Model - RMSE: {result["metrics"]["RMSE"]:.2f}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = f'MAE: {result["metrics"]["MAE"]:.2f} | MAPE: {result["metrics"]["MAPE"]:.1f}%'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=100, bbox_inches='tight')
    mlflow.log_artifact('predictions.png')
    plt.show()

if __name__ == "__main__":
    # Use shorter timeframe for better predictions
    results = run_analysis(
        symbol='AAPL',
        days_back=60,  # Use 60 days of history
        prediction_days=7,  # Predict 7 days ahead
        model_type='arima'  # Start with just ARIMA
    )
