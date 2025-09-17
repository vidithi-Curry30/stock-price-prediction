"""
Final validation of the complete system
"""
from src.predict import make_prediction
import json
from datetime import datetime

results = {}
for symbol in ['MSFT', 'AAPL', 'GOOGL']:
    pred = make_prediction(symbol)
    results[symbol] = {
        'date': pred['prediction_date'],
        'tradeable': pred['high_confidence_predictions'],
        'total': pred['total_predictions']
    }

# Save results
with open('results/final_test.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Final Test Complete")
print(f"Results saved to results/final_test.json")
for symbol, data in results.items():
    print(f"{symbol}: {data['tradeable']}/{data['total']} tradeable predictions")
