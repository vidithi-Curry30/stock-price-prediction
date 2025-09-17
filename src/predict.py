"""
Main prediction interface using VADPS-enhanced predictions
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.enhanced_prediction import EnhancedPredictor

def make_prediction(symbol, days=7, confidence_threshold=0.65):
    """
    Make predictions with VADPS enhancement
    
    Returns only high-confidence predictions suitable for trading
    """
    predictor = EnhancedPredictor()
    all_predictions = predictor.predict_with_vadps(symbol, days)
    
    # Filter for high-confidence predictions
    high_confidence = [p for p in all_predictions if p['confidence'] >= confidence_threshold]
    
    # Create summary
    summary = {
        'symbol': symbol,
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'total_predictions': len(all_predictions),
        'high_confidence_predictions': len(high_confidence),
        'predictions': high_confidence
    }
    
    return summary

def display_predictions(symbol):
    """Display predictions in a user-friendly format"""
    result = make_prediction(symbol)
    
    print(f"\nPredictions for {result['symbol']}")
    print(f"Date: {result['prediction_date']}")
    print(f"High-confidence predictions: {result['high_confidence_predictions']}/{result['total_predictions']}")
    
    if result['predictions']:
        print("\n📊 Actionable Predictions (>65% confidence):")
        print("-" * 50)
        for pred in result['predictions']:
            print(f"Day {pred['day']}: {pred['direction']} to ${pred['price']:.2f}")
            print(f"  Confidence: {pred['confidence']*100:.1f}%")
            print(f"  Range: ${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f}")
    else:
        print("\n⚠️ No high-confidence predictions available")
        print("Recommendation: Wait for better setup")

if __name__ == "__main__":
    # Test on multiple stocks
    for symbol in ['MSFT', 'AAPL', 'GOOGL']:
        display_predictions(symbol)
