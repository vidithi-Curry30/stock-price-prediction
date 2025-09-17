import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class ARIMAGARCHModel:
    def __init__(self):
        self.arima_model = None
        self.garch_model = None
        self.residuals = None
        self.order = None
        
    def analyze_residuals(self, residuals):
        """Perform comprehensive residual analysis"""
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'ljung_box': acorr_ljungbox(residuals, lags=10, return_df=True)
        }
        
        # Check for white noise (good if p-values > 0.05)
        lb_pvalues = analysis['ljung_box']['lb_pvalue'].values
        analysis['is_white_noise'] = all(p > 0.05 for p in lb_pvalues[:5])
        
        return analysis
    
    def find_optimal_order(self, data):
        """Find best ARIMA order using AIC/BIC"""
        best_aic = np.inf
        best_order = None
        
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        model = ARIMA(data, order=(p,d,q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p,d,q)
                    except:
                        continue
        
        self.order = best_order
        return best_order
    
    def train(self, data):
        """Train ARIMA model and GARCH on residuals"""
        
        # Step 1: Find optimal ARIMA order
        if self.order is None:
            self.order = self.find_optimal_order(data)
            print(f"Optimal ARIMA order: {self.order}")
        
        # Step 2: Fit ARIMA
        self.arima_model = ARIMA(data, order=self.order)
        self.arima_fitted = self.arima_model.fit()
        
        # Step 3: Analyze residuals
        self.residuals = self.arima_fitted.resid
        residual_analysis = self.analyze_residuals(self.residuals)
        
        print(f"Residual Analysis:")
        print(f"  Mean: {residual_analysis['mean']:.6f} (should be ~0)")
        print(f"  Std: {residual_analysis['std']:.4f}")
        print(f"  White noise test: {'PASS' if residual_analysis['is_white_noise'] else 'FAIL'}")
        
        # Step 4: Fit GARCH on squared residuals for volatility
        self.garch_model = arch_model(
            self.residuals.dropna(), 
            vol='GARCH', 
            p=1, 
            q=1,
            rescale=True
        )
        self.garch_fitted = self.garch_model.fit(disp='off')
        
        return residual_analysis
    
    def predict(self, steps=7):
        """Predict with volatility-adjusted confidence intervals"""
        
        # ARIMA predictions
        arima_forecast = self.arima_fitted.forecast(steps=steps)
        
        # GARCH volatility forecast
        garch_forecast = self.garch_fitted.forecast(horizon=steps)
        conditional_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
        
        # Adjust predictions based on volatility
        predictions = {
            'mean': arima_forecast.values,
            'volatility': conditional_volatility,
            'lower_95': arima_forecast.values - 1.96 * conditional_volatility,
            'upper_95': arima_forecast.values + 1.96 * conditional_volatility,
            'lower_99': arima_forecast.values - 2.58 * conditional_volatility,
            'upper_99': arima_forecast.values + 2.58 * conditional_volatility
        }
        
        return predictions

# Note: I chose GARCH(1,1) after testing various configurations.
# Higher orders didn't improve accuracy enough to justify complexity.
# The residual analysis was key - without validating white noise,
# the confidence intervals were way off.
