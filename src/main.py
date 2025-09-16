"""
Main entry point for stock price prediction framework
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import StockDataProcessor
from src.models.arima_model import ARIMAModel
from src.evaluation.metrics import ModelEvaluator
import argparse

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol')
    parser.add_argument('--period', default='5y', help='Time period')
    args = parser.parse_args()
    
    print(f"Starting analysis for {args.symbol}")
    
    # Initialize components
    data_processor = StockDataProcessor(args.symbol, args.period)
    arima_model = ARIMAModel()
    evaluator = ModelEvaluator()
    
    # Run analysis
    print("1. Fetching data...")
    data = data_processor.fetch_data()
    data = data_processor.add_technical_indicators()
    
    print("2. Training ARIMA model...")
    train_size = int(0.8 * len(data))
    train_data = data['Close'][:train_size]
    test_data = data['Close'][train_size:]
    
    fitted_model = arima_model.fit(train_data)
    predictions, _ = arima_model.predict(len(test_data))
    
    print("3. Evaluating model...")
    metrics = evaluator.calculate_metrics(test_data, predictions)
    print(f"Model Metrics: {metrics}")
    
    evaluator.plot_predictions(test_data, predictions, f"{args.symbol} ARIMA Predictions")

if __name__ == "__main__":
    main()
