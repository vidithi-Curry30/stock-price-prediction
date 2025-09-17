"""
Quick demo of the prediction system
Run this to see the system in action
"""

import sys
from src.predict import display_predictions

if len(sys.argv) > 1:
    symbol = sys.argv[1].upper()
else:
    symbol = 'MSFT'

print(f"\n{'='*60}")
print(f"Stock Price Prediction System Demo")
print(f"Using ARIMA-GARCH with VADPS Enhancement")
print(f"{'='*60}")

display_predictions(symbol)

print(f"\n{'='*60}")
print("System only shows predictions with >65% confidence")
print("This reduces overtrading and focuses on quality setups")
print(f"{'='*60}")
