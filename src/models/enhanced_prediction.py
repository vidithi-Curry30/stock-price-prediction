"""
Enhanced prediction system integrating VADPS with ARIMA-GARCH
This combines all models for superior predictions
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.arima_garch_model import ARIMAGARCHModel
from src.models.vadps_model import VADPSModel

class EnhancedPredictor:
    def __init__(self):
        self.arima_garch = ARIMAGARCHModel()
        self.vadps = VADPSModel()
        
    def predict_with_vadps(self, symbol, days_ahead=7):
        """
        Make predictions using ARIMA-GARCH enhanced with VADPS
        """
        print(f"Fetching data for {symbol}...")
        # Get data
        data = yf.Ticker(symbol).history(period='6mo')['Close']
        
        print("Training ARIMA-GARCH model...")
        # Train ARIMA-GARCH
        self.arima_garch.train(data[:-30])
        
        print("Calculating historical errors...")
        # Get historical errors for VADPS calibration
        historical_errors = self._calculate_historical_errors(data)
        
        # Generate base predictions
        base_predictions = self.arima_garch.predict(days_ahead)
        
        print("Applying VADPS enhancement...")
        # Enhance with VADPS
        vadps_scores = self.vadps.calculate_vadps(
            base_predictions,
            base_predictions['volatility'],
            historical_errors
        )
        
        # Combine into final predictions
        enhanced_predictions = self._combine_predictions(
            base_predictions, 
            vadps_scores,
            data.iloc[-1]
        )
        
        return enhanced_predictions
    
    def _calculate_historical_errors(self, data, lookback=10):
        """Calculate historical prediction errors for VADPS"""
        errors = []
        for i in range(lookback, 0, -1):
            if len(data) - i - 7 < 60:
                continue
            train = data[:-i-7]
            actual = data[-i]
            
            temp_model = ARIMAGARCHModel()
            temp_model.order = self.arima_garch.order
            temp_model.train(train)
            pred = temp_model.predict(7)
            
            errors.append(actual - pred['mean'][6])
        
        return np.array(errors) if errors else np.array([0])
    
    def _combine_predictions(self, base_predictions, vadps_scores, last_price):
        """
        Combine ARIMA-GARCH with VADPS for final predictions
        """
        enhanced = []
        current_price = last_price
        
        for i in range(len(vadps_scores)):
            # Base prediction from ARIMA
            base_price = base_predictions['mean'][i]
            
            # VADPS probability and direction
            vadps_prob = vadps_scores[i]['probability']
            vadps_direction = vadps_scores[i]['direction']
            
            # Adjust prediction based on VADPS confidence
            if vadps_prob > 0.65:  # High confidence
                # Keep prediction as is
                final_price = base_price
            else:  # Low confidence
                # Move prediction closer to current price
                adjustment = (base_price - current_price) * vadps_prob
                final_price = current_price + adjustment
            
            # Adjust confidence intervals based on VADPS
            band_adjustment = 2 - vadps_prob  # Wider bands for lower confidence
            lower_bound = final_price - vadps_scores[i]['confidence_band'] * band_adjustment
            upper_bound = final_price + vadps_scores[i]['confidence_band'] * band_adjustment
            
            enhanced.append({
                'day': i + 1,
                'price': final_price,
                'direction': 'UP' if vadps_direction > 0 else 'DOWN',
                'confidence': vadps_prob,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'tradeable': vadps_prob > 0.65  # Flag for actionable predictions
            })
            
            current_price = final_price
        
        return enhanced

def test_enhanced_predictions():
    """Test the enhanced prediction system"""
    predictor = EnhancedPredictor()
    
    for symbol in ['MSFT', 'AAPL']:
        print(f"\n{'='*60}")
        print(f"Enhanced Predictions for {symbol}")
        print('='*60)
        
        try:
            predictions = predictor.predict_with_vadps(symbol, days_ahead=7)
            
            # Display results
            print("\nDay | Price  | Direction | Confidence | Tradeable")
            print("----|--------|-----------|------------|----------")
            for pred in predictions[:5]:
                trade = "YES" if pred['tradeable'] else "NO"
                print(f"{pred['day']:3} | ${pred['price']:.2f} | {pred['direction']:9} | {pred['confidence']*100:9.1f}% | {trade}")
            
            # Summary statistics
            tradeable_days = [p for p in predictions if p['tradeable']]
            print(f"\nTradeable predictions: {len(tradeable_days)}/{len(predictions)}")
            if tradeable_days:
                avg_confidence = np.mean([p['confidence'] for p in tradeable_days])
                print(f"Average confidence on tradeable days: {avg_confidence*100:.1f}%")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    test_enhanced_predictions()
