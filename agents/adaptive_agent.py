#!/usr/bin/env python3
"""
Enhanced Adaptive Agent with Volatility-Based Range Adjustment
"""

import numpy as np
from typing import Dict, Any
from agents.agent_base import AgentBase

class AdaptiveAgent(AgentBase):
    """
    Adaptive agent that adjusts position ranges based on market volatility
    """
    
    def __init__(self, agent_id: str, initial_capital: float = 1.0, 
                 base_range_width: float = 0.1, volatility_multiplier: float = 2.0,
                 rebalance_threshold: float = 0.8):
        super().__init__(agent_id, initial_capital)
        
        # Strategy parameters
        self.base_range_width = base_range_width  # Base range width (10%)
        self.volatility_multiplier = volatility_multiplier  # Volatility scaling factor
        self.rebalance_threshold = rebalance_threshold  # When to rebalance (80% of range)
        
        # Adaptive parameters
        self.min_range_width = 0.05  # Minimum 5% range
        self.max_range_width = 0.3   # Maximum 30% range
        self.volatility_window = 24  # Hours to calculate volatility
        
        print("Adaptive Agent initialized with base_range_width={:.1%}, volatility_multiplier={:.1f}".format(
            self.base_range_width, self.volatility_multiplier))
    
    def decide_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide action based on current market conditions and volatility
        """
        current_price = market_data['price']
        current_volatility = self.get_recent_volatility(self.volatility_window)
        
        # Calculate adaptive range width based on volatility
        adaptive_range_width = self._calculate_adaptive_range_width(current_volatility)
        
        # If not in position, enter with adaptive range
        if not self.in_position:
            return self._enter_position(current_price, adaptive_range_width, current_volatility)
        
        # Check if current price is still within acceptable range
        range_position = self._get_range_position(current_price)
        
        # Rebalance if price is near range boundaries
        if abs(range_position) > self.rebalance_threshold:
            return self._rebalance_position(current_price, adaptive_range_width, 
                                          current_volatility, range_position)
        
        # Check if range width needs adjustment due to volatility change
        current_range_width = self.position_range[1] - self.position_range[0]
        if abs(current_range_width - adaptive_range_width) > 0.02:  # 2% difference
            return self._adjust_range_width(current_price, adaptive_range_width, current_volatility)
        
        # Hold position
        return {
            'action': 'hold',
            'reason': 'Price within range, volatility stable',
            'confidence': 0.7,
            'current_range_position': range_position,
            'adaptive_range_width': adaptive_range_width
        }
    
    def _calculate_adaptive_range_width(self, volatility: float) -> float:
        """Calculate range width based on current volatility"""
        # Scale range width by volatility
        adaptive_width = self.base_range_width + (volatility * self.volatility_multiplier)
        
        # Clamp to min/max bounds
        adaptive_width = max(self.min_range_width, min(self.max_range_width, adaptive_width))
        
        return adaptive_width
    
    def _get_range_position(self, current_price: float) -> float:
        """
        Get position within current range (-1 to 1)
        -1 = at lower bound, 0 = center, 1 = at upper bound
        """
        if not self.position_range:
            return 0.0
        
        range_center = (self.position_range[0] + self.position_range[1]) / 2
        range_half_width = (self.position_range[1] - self.position_range[0]) / 2
        
        # Normalize price position within range
        price_ratio = current_price / self.current_price  # Relative to entry price
        position = (price_ratio - range_center) / range_half_width
        
        return np.clip(position, -1.5, 1.5)  # Allow slight overflow
    
    def _enter_position(self, current_price: float, range_width: float, volatility: float) -> Dict[str, Any]:
        """Enter new position with adaptive range"""
        # Center range around current price
        lower_bound = 1.0 - (range_width / 2)
        upper_bound = 1.0 + (range_width / 2)
        
        return {
            'action': 'enter',
            'reason': 'Entering position with adaptive range (vol={:.2%})'.format(volatility),
            'new_range': (lower_bound, upper_bound),
            'confidence': 0.8,
            'adaptive_range_width': range_width,
            'volatility': volatility
        }
    
    def _rebalance_position(self, current_price: float, range_width: float, 
                          volatility: float, range_position: float) -> Dict[str, Any]:
        """Rebalance position due to price movement"""
        # Determine rebalancing strategy based on price movement direction
        if range_position > 0:
            # Price moved up, shift range up slightly
            shift_factor = 0.3  # Shift 30% towards price movement
            lower_bound = 1.0 - (range_width / 2) + (shift_factor * range_width / 2)
            upper_bound = 1.0 + (range_width / 2) + (shift_factor * range_width / 2)
            reason = "Price near upper bound, shifting range up"
        else:
            # Price moved down, shift range down slightly
            shift_factor = 0.3
            lower_bound = 1.0 - (range_width / 2) - (shift_factor * range_width / 2)
            upper_bound = 1.0 + (range_width / 2) - (shift_factor * range_width / 2)
            reason = "Price near lower bound, shifting range down"
        
        return {
            'action': 'rebalance',
            'reason': reason,
            'new_range': (lower_bound, upper_bound),
            'confidence': 0.9,
            'range_position': range_position,
            'adaptive_range_width': range_width,
            'volatility': volatility
        }
    
    def _adjust_range_width(self, current_price: float, new_range_width: float, 
                          volatility: float) -> Dict[str, Any]:
        """Adjust range width due to volatility change"""
        # Keep range centered around current relative price
        current_range_center = (self.position_range[0] + self.position_range[1]) / 2
        
        lower_bound = current_range_center - (new_range_width / 2)
        upper_bound = current_range_center + (new_range_width / 2)
        
        old_width = self.position_range[1] - self.position_range[0]
        width_change = "expanding" if new_range_width > old_width else "contracting"
        
        return {
            'action': 'rebalance',
            'reason': 'Adjusting range width due to volatility change ({})'.format(width_change),
            'new_range': (lower_bound, upper_bound),
            'confidence': 0.6,
            'old_range_width': old_width,
            'new_range_width': new_range_width,
            'volatility': volatility
        }
    
    def _is_profitable_after_costs(self, action: Dict[str, Any], gas_cost: float, fees: float) -> bool:
        """Enhanced profitability check for adaptive strategy"""
        total_cost = gas_cost + fees
        
        # More sophisticated profitability analysis
        if action['action'] == 'enter':
            # Always enter if costs are reasonable
            return total_cost < 0.02  # 2% threshold for entering
        
        elif action['action'] == 'rebalance':
            # Consider range position and volatility for rebalancing
            range_position = action.get('range_position', 0)
            volatility = action.get('volatility', 0.05)
            
            # Higher volatility justifies higher costs
            max_cost_threshold = 0.01 + (volatility * 0.5)  # Base 1% + volatility adjustment
            
            # Urgent rebalancing (near boundaries) justifies higher costs
            if abs(range_position) > 0.9:
                max_cost_threshold *= 1.5
            
            return total_cost < max_cost_threshold
        
        return super()._is_profitable_after_costs(action, gas_cost, fees)
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring"""
        current_volatility = self.get_recent_volatility(self.volatility_window)
        adaptive_range_width = self._calculate_adaptive_range_width(current_volatility)
        
        state = {
            'agent_type': 'adaptive',
            'current_volatility': current_volatility,
            'adaptive_range_width': adaptive_range_width,
            'base_range_width': self.base_range_width,
            'volatility_multiplier': self.volatility_multiplier,
            'in_position': self.in_position,
            'position_range': self.position_range
        }
        
        if self.in_position and self.current_price:
            state['range_position'] = self._get_range_position(self.current_price)
        
        return state 