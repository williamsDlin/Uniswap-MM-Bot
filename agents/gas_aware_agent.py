#!/usr/bin/env python3
"""
Enhanced Gas-Aware Agent with Percentile-Based Gas Optimization
"""

import numpy as np
from typing import Dict, Any, List
from agents.agent_base import AgentBase

class GasAwareAgent(AgentBase):
    """
    Gas-aware agent that delays transactions when gas prices are high
    Uses rolling percentiles to determine optimal transaction timing
    """
    
    def __init__(self, agent_id: str, initial_capital: float = 1.0,
                 gas_percentile_threshold: int = 80, gas_window: int = 24,
                 max_delay_hours: int = 6, range_width: float = 0.15):
        super().__init__(agent_id, initial_capital)
        
        # Gas optimization parameters
        self.gas_percentile_threshold = gas_percentile_threshold  # 80th percentile
        self.gas_window = gas_window  # 24 hours rolling window
        self.max_delay_hours = max_delay_hours  # Maximum delay before forced execution
        self.range_width = range_width  # Position range width
        
        # Delay tracking
        self.pending_action = None
        self.delay_start_time = None
        self.delay_count = 0
        
        # Gas price tracking
        self.gas_savings = 0.0  # Total gas saved by waiting
        self.delayed_transactions = []
        
        print("Gas-Aware Agent initialized with {}th percentile threshold, {}-hour window".format(
            self.gas_percentile_threshold, self.gas_window))
    
    def decide_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide action considering gas price optimization
        """
        current_price = market_data['price']
        current_gas = market_data['gas_price']
        timestamp = market_data.get('timestamp')
        
        # Check if we have a pending delayed action
        if self.pending_action:
            return self._handle_pending_action(market_data)
        
        # Determine base action needed
        base_action = self._determine_base_action(market_data)
        
        # If action is hold, no gas optimization needed
        if base_action['action'] == 'hold':
            return base_action
        
        # Check if gas price is favorable for execution
        if self._is_gas_price_favorable(current_gas):
            return base_action
        
        # Gas price is high - consider delaying
        return self._consider_delay(base_action, market_data)
    
    def _determine_base_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what action would be taken without gas considerations"""
        current_price = market_data['price']
        
        # If not in position, enter
        if not self.in_position:
            return self._create_enter_action(current_price)
        
        # Check if rebalancing is needed
        if self._needs_rebalancing(current_price):
            return self._create_rebalance_action(current_price)
        
        # Hold position
        return {
            'action': 'hold',
            'reason': 'Position within acceptable range',
            'confidence': 0.7
        }
    
    def _create_enter_action(self, current_price: float) -> Dict[str, Any]:
        """Create enter position action"""
        lower_bound = 1.0 - (self.range_width / 2)
        upper_bound = 1.0 + (self.range_width / 2)
        
        return {
            'action': 'enter',
            'reason': 'Entering new position',
            'new_range': (lower_bound, upper_bound),
            'confidence': 0.8
        }
    
    def _create_rebalance_action(self, current_price: float) -> Dict[str, Any]:
        """Create rebalance action"""
        # Simple rebalancing - center around current price
        lower_bound = 1.0 - (self.range_width / 2)
        upper_bound = 1.0 + (self.range_width / 2)
        
        return {
            'action': 'rebalance',
            'reason': 'Rebalancing position range',
            'new_range': (lower_bound, upper_bound),
            'confidence': 0.8
        }
    
    def _needs_rebalancing(self, current_price: float) -> bool:
        """Check if position needs rebalancing"""
        if not self.position_range:
            return False
        
        # Calculate current price position relative to range
        range_center = (self.position_range[0] + self.position_range[1]) / 2
        range_half_width = (self.position_range[1] - self.position_range[0]) / 2
        
        price_ratio = current_price / self.current_price
        position = abs(price_ratio - range_center) / range_half_width
        
        # Rebalance if price is more than 70% towards range boundary
        return position > 0.7
    
    def _is_gas_price_favorable(self, current_gas: float) -> bool:
        """Check if current gas price is below threshold"""
        gas_threshold = self.get_gas_percentile(self.gas_window, self.gas_percentile_threshold)
        return current_gas <= gas_threshold
    
    def _consider_delay(self, base_action: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Consider delaying action due to high gas prices"""
        current_gas = market_data['gas_price']
        timestamp = market_data.get('timestamp')
        
        # Calculate urgency of the action
        urgency = self._calculate_action_urgency(base_action, market_data)
        
        # If action is urgent, execute despite high gas
        if urgency > 0.8:
            return {
                **base_action,
                'reason': base_action['reason'] + ' (urgent despite high gas)',
                'gas_override': True,
                'urgency': urgency
            }
        
        # Delay the action
        self.pending_action = base_action
        self.delay_start_time = timestamp
        self.delay_count += 1
        
        gas_threshold = self.get_gas_percentile(self.gas_window, self.gas_percentile_threshold)
        
        return {
            'action': 'delay',
            'reason': 'Delaying due to high gas price ({:.1f} > {:.1f} gwei)'.format(
                current_gas, gas_threshold),
            'pending_action': base_action['action'],
            'gas_price': current_gas,
            'gas_threshold': gas_threshold,
            'confidence': 0.6,
            'urgency': urgency
        }
    
    def _handle_pending_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle previously delayed action"""
        current_gas = market_data['gas_price']
        timestamp = market_data.get('timestamp')
        
        # Check if we've been delaying too long
        if self._has_delayed_too_long(timestamp):
            action = self.pending_action
            self.pending_action = None
            self.delay_start_time = None
            
            return {
                **action,
                'reason': action['reason'] + ' (forced execution after max delay)',
                'forced_execution': True,
                'delay_duration': self.max_delay_hours
            }
        
        # Check if gas price is now favorable
        if self._is_gas_price_favorable(current_gas):
            action = self.pending_action
            self.pending_action = None
            delay_duration = self._calculate_delay_duration(timestamp)
            self.delay_start_time = None
            
            # Calculate gas savings
            original_gas_threshold = self.get_gas_percentile(self.gas_window, self.gas_percentile_threshold)
            gas_saved = max(0, original_gas_threshold - current_gas)
            self.gas_savings += gas_saved
            
            self.delayed_transactions.append({
                'timestamp': timestamp,
                'delay_duration': delay_duration,
                'gas_saved': gas_saved,
                'action': action['action']
            })
            
            return {
                **action,
                'reason': action['reason'] + ' (executed after {:.1f}h delay)'.format(delay_duration),
                'delay_duration': delay_duration,
                'gas_saved': gas_saved,
                'optimized_execution': True
            }
        
        # Continue delaying
        delay_duration = self._calculate_delay_duration(timestamp)
        return {
            'action': 'delay',
            'reason': 'Continuing to delay due to high gas price',
            'pending_action': self.pending_action['action'],
            'delay_duration': delay_duration,
            'confidence': 0.5
        }
    
    def _calculate_action_urgency(self, action: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate urgency of action (0-1 scale)"""
        if action['action'] == 'enter':
            # Entering is usually not urgent
            return 0.3
        
        elif action['action'] == 'rebalance':
            # Check how far price is from range center
            if not self.position_range:
                return 0.5
            
            current_price = market_data['price']
            range_center = (self.position_range[0] + self.position_range[1]) / 2
            range_half_width = (self.position_range[1] - self.position_range[0]) / 2
            
            price_ratio = current_price / self.current_price
            distance_from_center = abs(price_ratio - range_center) / range_half_width
            
            # Urgency increases as price approaches range boundaries
            return min(1.0, distance_from_center)
        
        return 0.5  # Default urgency
    
    def _has_delayed_too_long(self, current_timestamp) -> bool:
        """Check if action has been delayed beyond maximum allowed time"""
        if not self.delay_start_time or not current_timestamp:
            return False
        
        # Simple hour-based delay calculation
        # In real implementation, would use proper timestamp comparison
        return self.delay_count > self.max_delay_hours
    
    def _calculate_delay_duration(self, current_timestamp) -> float:
        """Calculate how long action has been delayed"""
        if not self.delay_start_time:
            return 0.0
        
        # Simplified calculation - in real implementation would use timestamps
        return min(self.delay_count, self.max_delay_hours)
    
    def _is_profitable_after_costs(self, action: Dict[str, Any], gas_cost: float, fees: float) -> bool:
        """Enhanced profitability check considering gas optimization"""
        total_cost = gas_cost + fees
        
        # If this is an optimized execution (after delay), be more lenient
        if action.get('optimized_execution', False):
            return total_cost < 0.02  # 2% threshold for optimized executions
        
        # If this is a forced execution, accept higher costs
        if action.get('forced_execution', False):
            return total_cost < 0.03  # 3% threshold for forced executions
        
        # Standard profitability check
        return total_cost < 0.01  # 1% threshold for normal executions
    
    def get_gas_optimization_metrics(self) -> Dict[str, Any]:
        """Get gas optimization performance metrics"""
        total_delays = len(self.delayed_transactions)
        avg_delay_duration = 0.0
        total_gas_saved = self.gas_savings
        
        if self.delayed_transactions:
            avg_delay_duration = np.mean([t['delay_duration'] for t in self.delayed_transactions])
        
        return {
            'total_delays': total_delays,
            'avg_delay_duration': avg_delay_duration,
            'total_gas_saved': total_gas_saved,
            'gas_savings_per_delay': total_gas_saved / max(total_delays, 1),
            'current_pending': self.pending_action is not None,
            'delay_count': self.delay_count
        }
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring"""
        gas_threshold = self.get_gas_percentile(self.gas_window, self.gas_percentile_threshold)
        
        state = {
            'agent_type': 'gas_aware',
            'gas_percentile_threshold': self.gas_percentile_threshold,
            'gas_window': self.gas_window,
            'current_gas_threshold': gas_threshold,
            'pending_action': self.pending_action is not None,
            'delay_count': self.delay_count,
            'total_gas_savings': self.gas_savings,
            'in_position': self.in_position,
            'position_range': self.position_range
        }
        
        if self.pending_action:
            state['pending_action_type'] = self.pending_action['action']
        
        return state
    
    def reset(self):
        """Reset agent state including gas optimization tracking"""
        super().reset()
        self.pending_action = None
        self.delay_start_time = None
        self.delay_count = 0
        self.gas_savings = 0.0
        self.delayed_transactions = [] 