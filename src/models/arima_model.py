import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self, order=None):
        self.order = order
        self.model = None
        self.model_fit = None
        
    def find_best_params(self, data, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3)):
        """Grid search to find best ARIMA parameters"""
        best_aic = np.inf
        best_order = None
        
        # Generate all combinations of p, d, q
        pdq_combinations = list(itertools.product(p_range, d_range, q_range))
        
        print("Searching for best ARIMA parameters...")
        for order in pdq_combinations:
            try:
                model = ARIMA(data, order=order)
                model_fit = model.fit()
                
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
            except:
                continue
        
        print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        return best_order
    
    def train(self, train_data):
        """Train the ARIMA model"""
        # Find best parameters if not specified
        if self.order is None:
            self.order = self.find_best_params(train_data)
        
        # Fit the model
        self.model = ARIMA(train_data, order=self.order)
        self.model_fit = self.model.fit()
        print(f"ARIMA{self.order} model fitted successfully")
        
        return self.model_fit
    
    def predict(self, steps):
        """Make predictions for the specified number of steps"""
        if self.model_fit is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        predictions = self.model_fit.forecast(steps=steps)
        
        return predictions
    
    def get_metrics(self):
        """Get model performance metrics"""
        if self.model_fit is None:
            return None
            
        return {
            'AIC': self.model_fit.aic,
            'BIC': self.model_fit.bic,
            'Log_Likelihood': self.model_fit.llf
        }

def predict_with_confidence(self, steps, alpha=0.05):
    """Make predictions with confidence intervals"""
    forecast = self.model_fit.get_forecast(steps=steps)
    predictions = forecast.predicted_mean
    confidence_int = forecast.conf_int(alpha=alpha)
    return predictions, confidence_int
