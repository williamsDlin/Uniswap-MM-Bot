"""
Gas Price Prediction Model using SARIMA for transaction timing optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.base_agent import BaseAgent

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class GasPredictorAgent(BaseAgent):
    """
    SARIMA-based gas price predictor for optimizing transaction timing
    """
    
    def __init__(self, 
                 max_history: int = 1000,
                 prediction_horizon: int = 12,  # Hours ahead to predict
                 update_frequency: int = 10,    # Update model every N observations
                 agent_name: str = "gas_predictor"):
        """
        Initialize gas price predictor
        
        Args:
            max_history: Maximum number of historical data points to keep
            prediction_horizon: How many hours ahead to predict
            update_frequency: How often to retrain the model
            agent_name: Name for this agent
        """
        super().__init__(agent_name)
        
        self.max_history = max_history
        self.prediction_horizon = prediction_horizon
        self.update_frequency = update_frequency
        
        # Historical data storage
        self.gas_price_history = []  # List of (timestamp, gas_price_gwei)
        self.model = None
        self.model_fitted = False
        self.last_update_count = 0
        
        # Model parameters (will be tuned automatically)
        self.arima_order = (2, 1, 2)  # (p, d, q)
        self.seasonal_order = (1, 1, 1, 24)  # (P, D, Q, s) - 24h seasonality
        
        # Prediction cache
        self.cached_predictions = {}
        self.last_prediction_time = None
        
        # Performance tracking
        self.prediction_errors = []
        self.model_accuracy_metrics = {}
        
        logger.info(f"Gas predictor initialized with {prediction_horizon}h horizon")
    
    def add_gas_price_observation(self, gas_price_gwei: float, timestamp: datetime = None) -> None:
        """
        Add new gas price observation
        
        Args:
            gas_price_gwei: Gas price in Gwei
            timestamp: Timestamp of observation (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self.gas_price_history.append((timestamp, gas_price_gwei))
        
        # Maintain maximum history size
        if len(self.gas_price_history) > self.max_history:
            self.gas_price_history = self.gas_price_history[-self.max_history:]
        
        # Check if we should update the model
        if len(self.gas_price_history) - self.last_update_count >= self.update_frequency:
            self._update_model()
        
        logger.debug(f"Added gas price observation: {gas_price_gwei:.2f} Gwei")
    
    def _prepare_time_series_data(self) -> Optional[pd.Series]:
        """Prepare time series data for modeling"""
        if len(self.gas_price_history) < 48:  # Need at least 48 hours of data
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(self.gas_price_history, columns=['timestamp', 'gas_price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to hourly averages (in case we have more frequent data)
        hourly_data = df.resample('H').mean().fillna(method='ffill')
        
        # Remove outliers (gas prices > 500 Gwei are likely errors)
        hourly_data = hourly_data[hourly_data['gas_price'] <= 500]
        
        if len(hourly_data) < 48:
            return None
        
        return hourly_data['gas_price']
    
    def _auto_tune_parameters(self, ts_data: pd.Series) -> Tuple[tuple, tuple]:
        """
        Automatically tune ARIMA parameters using AIC
        
        Args:
            ts_data: Time series data
            
        Returns:
            Tuple of (best_order, best_seasonal_order)
        """
        try:
            best_aic = float('inf')
            best_order = self.arima_order
            best_seasonal_order = self.seasonal_order
            
            # Test different parameter combinations
            p_values = [0, 1, 2]
            d_values = [0, 1]
            q_values = [0, 1, 2]
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            order = (p, d, q)
                            seasonal_order = (1, 1, 1, 24)  # Keep seasonal fixed
                            
                            model = ARIMA(ts_data, order=order, seasonal_order=seasonal_order)
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = order
                                best_seasonal_order = seasonal_order
                                
                        except Exception:
                            continue
            
            logger.debug(f"Best ARIMA parameters: {best_order}, seasonal: {best_seasonal_order}, AIC: {best_aic:.2f}")
            return best_order, best_seasonal_order
            
        except Exception as e:
            logger.warning(f"Parameter tuning failed, using defaults: {e}")
            return self.arima_order, self.seasonal_order
    
    def _update_model(self) -> None:
        """Update the SARIMA model with new data"""
        try:
            # Prepare data
            ts_data = self._prepare_time_series_data()
            if ts_data is None:
                logger.debug("Insufficient data for model update")
                return
            
            # Auto-tune parameters periodically
            if len(self.gas_price_history) % 100 == 0:
                self.arima_order, self.seasonal_order = self._auto_tune_parameters(ts_data)
            
            # Fit SARIMA model
            model = ARIMA(
                ts_data, 
                order=self.arima_order,
                seasonal_order=self.seasonal_order
            )
            
            self.model = model.fit()
            self.model_fitted = True
            self.last_update_count = len(self.gas_price_history)
            
            # Calculate model accuracy on recent data
            if len(ts_data) > 24:
                # Use last 24 hours for validation
                train_data = ts_data[:-24]
                test_data = ts_data[-24:]
                
                # Refit on training data
                train_model = ARIMA(
                    train_data, 
                    order=self.arima_order,
                    seasonal_order=self.seasonal_order
                ).fit()
                
                # Make predictions
                predictions = train_model.forecast(steps=24)
                
                # Calculate accuracy metrics
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mean_squared_error(test_data, predictions))
                mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
                
                self.model_accuracy_metrics = {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'last_updated': datetime.now().isoformat()
                }
                
                logger.info(f"Model updated. Accuracy - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
            
            # Clear prediction cache
            self.cached_predictions.clear()
            
        except Exception as e:
            logger.error(f"Error updating gas prediction model: {e}")
    
    def predict_gas_prices(self, hours_ahead: int = None) -> Optional[Dict[int, float]]:
        """
        Predict gas prices for the next N hours
        
        Args:
            hours_ahead: Number of hours to predict (default: self.prediction_horizon)
            
        Returns:
            Dictionary mapping hours_ahead -> predicted_gas_price_gwei
        """
        if hours_ahead is None:
            hours_ahead = self.prediction_horizon
        
        if not self.model_fitted or self.model is None:
            logger.warning("Model not fitted, cannot make predictions")
            return None
        
        try:
            # Check cache
            cache_key = f"pred_{hours_ahead}"
            now = datetime.now()
            
            if (cache_key in self.cached_predictions and 
                self.last_prediction_time and 
                (now - self.last_prediction_time).total_seconds() < 3600):  # 1 hour cache
                return self.cached_predictions[cache_key]
            
            # Make predictions
            predictions = self.model.forecast(steps=hours_ahead)
            
            # Convert to dictionary
            prediction_dict = {}
            for i, pred in enumerate(predictions, 1):
                prediction_dict[i] = max(1.0, float(pred))  # Ensure minimum 1 Gwei
            
            # Cache results
            self.cached_predictions[cache_key] = prediction_dict
            self.last_prediction_time = now
            
            logger.debug(f"Predicted gas prices for next {hours_ahead} hours: {prediction_dict}")
            
            return prediction_dict
            
        except Exception as e:
            logger.error(f"Error making gas price predictions: {e}")
            return None
    
    def should_execute_now(self, 
                          current_gas_price: float,
                          max_gas_price: float = 50.0,
                          urgency_hours: int = 6) -> Tuple[bool, str, float]:
        """
        Determine if transaction should be executed now based on gas price forecasts
        
        Args:
            current_gas_price: Current gas price in Gwei
            max_gas_price: Maximum acceptable gas price
            urgency_hours: How many hours we can wait
            
        Returns:
            Tuple of (should_execute, reason, confidence_score)
        """
        try:
            # If current price is below threshold, execute immediately
            if current_gas_price <= max_gas_price:
                return True, f"Current gas price ({current_gas_price:.1f}) below threshold ({max_gas_price})", 1.0
            
            # If we have no model, be conservative
            if not self.model_fitted:
                if current_gas_price <= max_gas_price * 1.5:
                    return True, "No prediction model, gas price acceptable", 0.5
                else:
                    return False, "No prediction model, gas price too high", 0.5
            
            # Get predictions for urgency window
            predictions = self.predict_gas_prices(urgency_hours)
            if predictions is None:
                return True, "Prediction failed, executing conservatively", 0.3
            
            # Find minimum predicted price in the urgency window
            min_predicted_price = min(predictions.values())
            min_price_hour = min(predictions.keys(), key=lambda k: predictions[k])
            
            # Calculate potential savings
            potential_savings = current_gas_price - min_predicted_price
            savings_percent = potential_savings / current_gas_price * 100
            
            # Decision logic
            if min_predicted_price >= current_gas_price:
                # Prices expected to increase or stay same
                return True, f"Prices expected to increase (min: {min_predicted_price:.1f})", 0.8
            
            elif savings_percent > 20:
                # Significant savings possible (>20%)
                return False, f"Significant savings possible in {min_price_hour}h ({savings_percent:.1f}%)", 0.9
            
            elif savings_percent > 10 and urgency_hours > 3:
                # Moderate savings and we have time
                return False, f"Moderate savings possible in {min_price_hour}h ({savings_percent:.1f}%)", 0.7
            
            else:
                # Small savings or urgent
                return True, f"Small savings ({savings_percent:.1f}%) or time constraint", 0.6
                
        except Exception as e:
            logger.error(f"Error in gas execution decision: {e}")
            return True, "Error in prediction, executing conservatively", 0.3
    
    def get_optimal_execution_time(self, 
                                  max_hours: int = 24) -> Tuple[int, float, float]:
        """
        Get optimal execution time within the next N hours
        
        Args:
            max_hours: Maximum hours to look ahead
            
        Returns:
            Tuple of (optimal_hour, predicted_gas_price, confidence)
        """
        predictions = self.predict_gas_prices(max_hours)
        if predictions is None:
            return 1, 50.0, 0.0  # Default: execute in 1 hour at 50 Gwei
        
        # Find hour with minimum gas price
        optimal_hour = min(predictions.keys(), key=lambda k: predictions[k])
        optimal_price = predictions[optimal_hour]
        
        # Calculate confidence based on price variation
        price_values = list(predictions.values())
        price_std = np.std(price_values)
        price_range = max(price_values) - min(price_values)
        
        # Higher confidence if there's clear variation and we found a low point
        confidence = min(1.0, price_range / np.mean(price_values))
        
        return optimal_hour, optimal_price, confidence
    
    def select_action(self, context: np.ndarray) -> Any:
        """Required by base class - not used for gas predictor"""
        return None
    
    def update(self, context: np.ndarray, action: Any, reward: float) -> None:
        """Required by base class - gas predictor uses different update mechanism"""
        pass
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization"""
        return {
            'max_history': self.max_history,
            'prediction_horizon': self.prediction_horizon,
            'update_frequency': self.update_frequency,
            'arima_order': self.arima_order,
            'seasonal_order': self.seasonal_order,
            'gas_price_history': self.gas_price_history[-100:],  # Keep recent history
            'model_fitted': self.model_fitted,
            'model_accuracy_metrics': self.model_accuracy_metrics,
            'prediction_errors': self.prediction_errors[-50:]  # Keep recent errors
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loaded data"""
        self.max_history = state.get('max_history', self.max_history)
        self.prediction_horizon = state.get('prediction_horizon', self.prediction_horizon)
        self.update_frequency = state.get('update_frequency', self.update_frequency)
        self.arima_order = state.get('arima_order', self.arima_order)
        self.seasonal_order = state.get('seasonal_order', self.seasonal_order)
        self.gas_price_history = state.get('gas_price_history', [])
        self.model_fitted = state.get('model_fitted', False)
        self.model_accuracy_metrics = state.get('model_accuracy_metrics', {})
        self.prediction_errors = state.get('prediction_errors', [])
        
        # Retrain model if we have enough data
        if len(self.gas_price_history) >= 48:
            self._update_model()
    
    def get_gas_statistics(self) -> Dict[str, Any]:
        """Get gas price statistics and model performance"""
        if not self.gas_price_history:
            return {"status": "no_data"}
        
        recent_prices = [price for _, price in self.gas_price_history[-24:]]
        
        stats = {
            'total_observations': len(self.gas_price_history),
            'model_fitted': self.model_fitted,
            'recent_avg_gas': np.mean(recent_prices) if recent_prices else 0,
            'recent_min_gas': min(recent_prices) if recent_prices else 0,
            'recent_max_gas': max(recent_prices) if recent_prices else 0,
            'recent_std_gas': np.std(recent_prices) if recent_prices else 0,
            'model_accuracy': self.model_accuracy_metrics,
            'prediction_horizon': self.prediction_horizon
        }
        
        # Add current predictions if model is fitted
        if self.model_fitted:
            predictions = self.predict_gas_prices(6)  # Next 6 hours
            if predictions:
                stats['next_6h_predictions'] = predictions
                stats['predicted_min_6h'] = min(predictions.values())
                stats['predicted_max_6h'] = max(predictions.values())
        
        return stats 