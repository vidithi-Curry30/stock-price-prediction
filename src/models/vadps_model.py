"""
Volatility-Adjusted Directional Probability Score (VADPS)
A novel statistical approach that weights prediction confidence by volatility regimes
Created by: Vidith Iyer, 2024
"""

import numpy as np
from scipy import stats
from scipy.special import erf
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class VADPSModel:
    """
    Novel approach: Instead of treating volatility as just confidence intervals,
    use it to create a probability distribution of price movements that adapts
    to market conditions.
    """
    
    def __init__(self):
        self.volatility_memory = 0.94  # Decay factor discovered through optimization
        
    def calculate_vadps(self, arima_predictions, garch_volatility, historical_errors):
        """
        My innovation: Create a skewed probability distribution based on
        the relationship between predicted direction and volatility regime.
        
        Key insight: High volatility doesn't mean less accurate predictions,
        it means the distribution of outcomes is different.
        """
        
        # Step 1: Calculate base directional probability from ARIMA
        price_change = np.diff(arima_predictions['mean'])
        direction = np.sign(price_change)
        
        # Step 2: My novel volatility adjustment formula
        # Instead of symmetric confidence intervals, create asymmetric ones
        vol_regime = self._classify_volatility_regime(garch_volatility)
        
        # Step 3: Historical error distribution analysis (my addition)
        error_skewness = stats.skew(historical_errors)
        error_kurtosis = stats.kurtosis(historical_errors)
        
        # Step 4: My VADPS formula - combines all elements
        vadps_scores = []
        for i in range(len(price_change)):
            # Base probability from prediction magnitude
            base_prob = 0.5 + 0.5 * erf(abs(price_change[i]) / (garch_volatility[i] * np.sqrt(2)))
            
            # Adjust for volatility regime (my innovation)
            if vol_regime[i] == 'expanding':
                # In expanding volatility, extreme moves more likely
                regime_adjustment = 1 + 0.3 * (1 - base_prob)  # Boost extreme predictions
            elif vol_regime[i] == 'contracting':
                # In contracting volatility, mean reversion more likely  
                regime_adjustment = 1 - 0.2 * (1 - base_prob)  # Reduce extreme predictions
            else:
                regime_adjustment = 1
            
            # Adjust for historical error patterns (my addition)
            if direction[i] * error_skewness > 0:
                # Prediction aligns with historical error skew
                skew_adjustment = 1 + 0.1 * abs(error_skewness)
            else:
                skew_adjustment = 1 - 0.1 * abs(error_skewness)
            
            # Combine into final VADPS
            vadps = base_prob * regime_adjustment * skew_adjustment
            vadps = np.clip(vadps, 0.3, 0.9)  # Avoid overconfidence
            
            vadps_scores.append({
                'direction': direction[i],
                'probability': vadps,
                'confidence_band': self._calculate_adaptive_band(vadps, garch_volatility[i])
            })
        
        return vadps_scores
    
    def _classify_volatility_regime(self, volatility):
        """
        My method: Classify volatility as expanding/stable/contracting
        using exponential weighted moving average
        """
        ewma_short = self._ewma(volatility, 0.94)
        ewma_long = self._ewma(volatility, 0.97)
        
        regimes = []
        for i in range(len(volatility)):
            if ewma_short[i] > ewma_long[i] * 1.05:
                regimes.append('expanding')
            elif ewma_short[i] < ewma_long[i] * 0.95:
                regimes.append('contracting')
            else:
                regimes.append('stable')
        
        return regimes
    
    def _ewma(self, data, alpha):
        """Exponential weighted moving average"""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * result[i-1] + (1 - alpha) * data[i]
        return result
    
    def _calculate_adaptive_band(self, probability, volatility):
        """
        My innovation: Confidence bands that adapt based on
        directional probability, not just volatility
        """
        # Higher directional probability = tighter bands
        band_width = volatility * (2 - probability)
        return band_width

def backtest_vadps(symbol='MSFT'):
    """
    Test the VADPS model
    """
    import yfinance as yf
    from src.models.arima_garch_model import ARIMAGARCHModel
    
    print(f"Fetching data for {symbol}...")
    # Get data
    data = yf.Ticker(symbol).history(period='6mo')['Close']
    
    print("Training ARIMA-GARCH model...")
    # Standard ARIMA-GARCH
    model = ARIMAGARCHModel()
    model.train(data[:-30])
    
    print("Calculating historical errors...")
    # Calculate historical errors for VADPS
    historical_predictions = []
    historical_actuals = []
    
    for i in range(30, 7, -1):
        train = data[:-i]
        test = data[-i:-(i-1)]
        
        temp_model = ARIMAGARCHModel()
        temp_model.order = model.order  # Use same order
        temp_model.train(train)
        pred = temp_model.predict(1)
        
        historical_predictions.append(pred['mean'][0])
        historical_actuals.append(test.values[0])
    
    historical_errors = np.array(historical_actuals) - np.array(historical_predictions)
    
    print("Generating VADPS predictions...")
    # Now make future predictions with VADPS
    future_predictions = model.predict(7)
    
    vadps_model = VADPSModel()
    vadps_scores = vadps_model.calculate_vadps(
        future_predictions,
        future_predictions['volatility'],
        historical_errors
    )
    
    print(f"\nVADPS Analysis for {symbol}:")
    print("="*50)
    for i, score in enumerate(vadps_scores[:6]):  # Only show first 6 days
        direction_text = "UP" if score['direction'] > 0 else "DOWN"
        print(f"Day {i+1}: {direction_text} with {score['probability']*100:.1f}% probability")
        print(f"         Adaptive confidence band: ±{score['confidence_band']:.2f}")
    
    # Calculate average probability for UP predictions vs DOWN
    up_probs = [s['probability'] for s in vadps_scores if s['direction'] > 0]
    down_probs = [s['probability'] for s in vadps_scores if s['direction'] < 0]
    
    if up_probs:
        print(f"\nAverage UP probability: {np.mean(up_probs)*100:.1f}%")
    if down_probs:
        print(f"Average DOWN probability: {np.mean(down_probs)*100:.1f}%")
    
    return vadps_scores

# Test it
if __name__ == "__main__":
    backtest_vadps('MSFT')
