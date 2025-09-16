from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        
    def find_best_order(self, data, max_p=3, max_d=2, max_q=3):
        best_aic = float('inf')
        best_order = None
        
        print("Searching for best ARIMA parameters...")
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        return best_order
    
    def fit(self, train_data, order=None):
        if order is None:
            order = self.find_best_order(train_data)
        
        self.model = ARIMA(train_data, order=order)
        self.fitted_model = self.model.fit()
        print(f"ARIMA{order} model fitted successfully")
        return self.fitted_model
    
    def predict(self, steps):
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        forecast = self.fitted_model.forecast(steps=steps)
        conf_int = self.fitted_model.get_forecast(steps=steps).conf_int()
        return forecast, conf_int
