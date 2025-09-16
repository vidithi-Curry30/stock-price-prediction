import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    @staticmethod
    def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual', color='blue')
        plt.plot(y_true.index, y_pred, label='Predicted', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
