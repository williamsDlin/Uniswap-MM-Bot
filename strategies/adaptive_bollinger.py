"""
Adaptive Bollinger Bands strategy using LinUCB contextual bandit
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta

from strategies.bollinger import BollingerBandsStrategy
from ml.bandit_strategy import LinUCBBanditAgent

logger = logging.getLogger(__name__)

class AdaptiveBollingerBandsStrategy(BollingerBandsStrategy):
    """
    Enhanced Bollinger Bands strategy with adaptive parameter selection using LinUCB
    """
    
    def __init__(self, 
                 initial_window: int = 20, 
                 initial_std_dev: float = 2.0,
                 bandit_alpha: float = 1.0,
                 reward_lookback_periods: int = 5,
                 min_learning_periods: int = 10):
        """
        Initialize adaptive Bollinger Bands strategy
        
        Args:
            initial_window: Initial window size (used until agent learns)
            initial_std_dev: Initial standard deviation (used until agent learns)
            bandit_alpha: LinUCB exploration parameter
            reward_lookback_periods: Number of periods to look back for reward calculation
            min_learning_periods: Minimum periods before using bandit decisions
        """
        # Initialize with initial parameters
        super().__init__(window=initial_window, std_dev=initial_std_dev)
        
        # Initialize bandit agent
        self.bandit = LinUCBBanditAgent(alpha=bandit_alpha)
        self.bandit.load_model()  # Try to load existing model
        
        self.reward_lookback_periods = reward_lookback_periods
        self.min_learning_periods = min_learning_periods
        
        # Enhanced tracking for reward calculation
        self.position_history = []  # Track position performance
        self.rebalance_history = []  # Track rebalance events
        self.parameter_history = []  # Track parameter choices
        self.context_history = []  # Track context features
        
        # Current active parameters
        self.current_window = initial_window
        self.current_std_dev = initial_std_dev
        self.last_parameter_update = None
        self.last_context = None
        
        logger.info(f"Adaptive Bollinger strategy initialized with bandit alpha={bandit_alpha}")
    
    def add_price(self, price: float, 
                  position_value: Optional[float] = None,
                  fees_earned: Optional[float] = None,
                  gas_cost: Optional[float] = None) -> None:
        """
        Add a new price and optionally update performance metrics
        
        Args:
            price: New price observation
            position_value: Current position value (for reward calculation)
            fees_earned: Fees earned since last update
            gas_cost: Gas costs incurred since last update
        """
        # Add price to base strategy
        super().add_price(price)
        
        # Record position performance if provided
        if position_value is not None:
            self.position_history.append({
                'timestamp': datetime.now(),
                'price': price,
                'position_value': position_value,
                'fees_earned': fees_earned or 0.0,
                'gas_cost': gas_cost or 0.0,
                'window': self.current_window,
                'std_dev': self.current_std_dev
            })
        
        # Update parameters if we have enough data and it's time to adapt
        if self._should_update_parameters():
            self._update_parameters()
    
    def _should_update_parameters(self) -> bool:
        """Determine if we should update parameters"""
        # Need minimum number of price observations
        if len(self.price_history) < self.min_learning_periods:
            return False
        
        # Update every 5 price observations or if we have a new position
        if len(self.position_history) == 0:
            return True
        
        return len(self.price_history) % 5 == 0
    
    def _update_parameters(self) -> None:
        """Update Bollinger Band parameters using the bandit"""
        try:
            # Extract context from current market conditions
            context = self.bandit.extract_context(self.price_history)
            
            # Get parameter recommendation from bandit
            new_window, new_std_dev = self.bandit.select_action(context)
            
            # Calculate reward for previous parameter choice if applicable
            if (self.last_context is not None and 
                self.last_parameter_update is not None and
                len(self.position_history) > 0):
                
                reward = self._calculate_reward()
                self.bandit.update(
                    self.last_context, 
                    (self.current_window, self.current_std_dev), 
                    reward
                )
            
            # Update parameters
            old_params = (self.current_window, self.current_std_dev)
            self.current_window = new_window
            self.current_std_dev = new_std_dev
            
            # Update base strategy parameters
            self.window = new_window
            self.std_dev = new_std_dev
            
            # Record parameter change
            self.parameter_history.append({
                'timestamp': datetime.now(),
                'old_window': old_params[0],
                'old_std_dev': old_params[1],
                'new_window': new_window,
                'new_std_dev': new_std_dev,
                'context': context.tolist(),
                'reward': self._calculate_reward() if self.position_history else 0.0
            })
            
            # Store for next update
            self.last_context = context
            self.last_parameter_update = datetime.now()
            
            logger.info(f"Updated parameters: window {old_params[0]}→{new_window}, "
                       f"std_dev {old_params[1]:.2f}→{new_std_dev:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on recent position performance
        
        Returns:
            Reward value (higher is better)
        """
        if len(self.position_history) < 2:
            return 0.0
        
        try:
            # Look at recent performance (last N periods)
            recent_history = self.position_history[-self.reward_lookback_periods:]
            
            if len(recent_history) < 2:
                return 0.0
            
            # Calculate total fees earned and gas costs
            total_fees = sum([h['fees_earned'] for h in recent_history])
            total_gas = sum([h['gas_cost'] for h in recent_history])
            
            # Calculate position value change
            start_value = recent_history[0]['position_value']
            end_value = recent_history[-1]['position_value']
            value_change = end_value - start_value if start_value > 0 else 0
            
            # Calculate price range efficiency (how much of the time price was in range)
            in_range_count = 0
            total_periods = len(recent_history) - 1
            
            for i in range(1, len(recent_history)):
                hist = recent_history[i]
                # Estimate if price was likely in range based on strategy logic
                bands = self._estimate_bands_for_period(hist)
                if bands and bands[0] <= hist['price'] <= bands[2]:
                    in_range_count += 1
            
            range_efficiency = in_range_count / total_periods if total_periods > 0 else 0.5
            
            # Reward components:
            # 1. Net profit (fees - gas)
            net_profit = total_fees - total_gas
            
            # 2. Range efficiency bonus
            efficiency_bonus = range_efficiency * 0.001  # Small bonus for good range selection
            
            # 3. Stability bonus (penalty for too frequent rebalancing)
            rebalance_penalty = len([h for h in recent_history if 'rebalance' in str(h)]) * 0.0001
            
            # Total reward
            reward = net_profit + efficiency_bonus - rebalance_penalty
            
            logger.debug(f"Reward calculation: profit={net_profit:.6f}, "
                        f"efficiency={efficiency_bonus:.6f}, penalty={rebalance_penalty:.6f}, "
                        f"total={reward:.6f}")
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _estimate_bands_for_period(self, hist_record: Dict) -> Optional[Tuple[float, float, float]]:
        """Estimate what the Bollinger bands were for a historical period"""
        try:
            # This is a simplified estimation - in practice you'd want to store actual bands
            window = hist_record.get('window', self.window)
            std_dev = hist_record.get('std_dev', self.std_dev)
            
            # Find the price index in history
            target_price = hist_record['price']
            price_idx = None
            
            for i, price in enumerate(self.price_history):
                if abs(price - target_price) < 0.01:  # Small tolerance
                    price_idx = i
                    break
            
            if price_idx is None or price_idx < window:
                return None
            
            # Calculate bands for that period
            prices = self.price_history[price_idx-window:price_idx]
            if len(prices) < window:
                return None
            
            prices_series = pd.Series(prices)
            middle = prices_series.mean()
            std = prices_series.std()
            
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            return lower, middle, upper
            
        except Exception:
            return None
    
    def record_rebalance_event(self, 
                              old_range: Optional[Tuple[float, float]] = None,
                              new_range: Optional[Tuple[float, float]] = None,
                              gas_cost: float = 0.0,
                              reason: str = "") -> None:
        """Record a rebalancing event for performance tracking"""
        self.rebalance_history.append({
            'timestamp': datetime.now(),
            'old_range': old_range,
            'new_range': new_range,
            'gas_cost': gas_cost,
            'reason': reason,
            'price': self.price_history[-1] if self.price_history else 0,
            'window': self.current_window,
            'std_dev': self.current_std_dev
        })
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including bandit performance"""
        base_stats = super().get_statistics()
        
        # Add bandit statistics
        bandit_stats = self.bandit.get_stats()
        action_stats = self.bandit.get_action_stats()
        
        # Add adaptive strategy specific stats
        adaptive_stats = {
            'current_parameters': {
                'window': self.current_window,
                'std_dev': self.current_std_dev
            },
            'parameter_changes': len(self.parameter_history),
            'rebalance_events': len(self.rebalance_history),
            'position_records': len(self.position_history),
            'bandit_performance': bandit_stats,
            'action_distribution': action_stats
        }
        
        # Calculate recent performance
        if len(self.position_history) >= 2:
            recent_positions = self.position_history[-10:]
            total_fees = sum([p['fees_earned'] for p in recent_positions])
            total_gas = sum([p['gas_cost'] for p in recent_positions])
            
            adaptive_stats['recent_performance'] = {
                'total_fees': total_fees,
                'total_gas': total_gas,
                'net_profit': total_fees - total_gas,
                'average_reward': np.mean([p.get('reward', 0) for p in self.parameter_history[-10:]])
            }
        
        # Combine all statistics
        combined_stats = {**base_stats, **adaptive_stats}
        
        return combined_stats
    
    def save_model(self) -> None:
        """Save the bandit model"""
        self.bandit.save_model()
        logger.info("Saved adaptive Bollinger strategy model")
    
    def should_rebalance(self, 
                        current_price: float, 
                        current_range: Optional[Tuple[float, float]] = None) -> Tuple[bool, str]:
        """
        Enhanced rebalancing logic that considers adaptive parameters
        """
        # Use the base strategy logic but with current adaptive parameters
        return super().should_rebalance(current_price, current_range)
    
    def get_optimal_range(self, current_price: float) -> Tuple[float, float]:
        """
        Get optimal range using current adaptive parameters
        """
        return super().get_optimal_range(current_price) 