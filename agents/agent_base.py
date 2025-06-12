#!/usr/bin/env python3
"""
Base Agent Interface for Uniswap V3 Multi-Agent Framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

class AgentBase(ABC):
    """
    Abstract base class for all trading agents
    """
    
    def __init__(self, agent_id: str, initial_capital: float = 1.0):
        self.agent_id = agent_id
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_history = []
        self.transaction_history = []
        self.performance_metrics = {}
        
        # Market data access
        self.current_price = None
        self.current_gas_price = None
        self.price_history = []
        self.gas_history = []
        self.volume_history = []
        
        # Position state
        self.in_position = False
        self.position_range = None  # (lower_tick, upper_tick)
        self.position_liquidity = 0.0
        
        print("Initialized agent: {}".format(self.agent_id))
    
    @abstractmethod
    def decide_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main decision function - must be implemented by each agent
        
        Args:
            market_data: Dict containing current market state
                - price: current ETH price
                - gas_price: current gas price in gwei
                - volume: current trading volume
                - volatility: recent price volatility
                - liquidity_depth: pool liquidity
                - timestamp: current timestamp
        
        Returns:
            Dict with action decision:
                - action: 'hold', 'rebalance', 'enter', 'exit', 'delay'
                - reason: explanation for the decision
                - new_range: (lower, upper) if rebalancing
                - confidence: 0-1 confidence score
        """
        pass
    
    def update_market_data(self, market_data: Dict[str, Any]):
        """Update agent's view of market conditions"""
        self.current_price = market_data['price']
        self.current_gas_price = market_data['gas_price']
        
        # Update histories
        self.price_history.append(market_data['price'])
        self.gas_history.append(market_data['gas_price'])
        self.volume_history.append(market_data.get('volume', 0))
        
        # Keep only recent history (last 168 hours = 1 week)
        max_history = 168
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.gas_history = self.gas_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
    
    def execute_action(self, action: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the decided action and update agent state
        
        Returns:
            Dict with execution results:
                - executed: bool
                - gas_cost: float
                - fees: float
                - new_capital: float
        """
        action_type = action['action']
        gas_cost = 0.0
        fees = 0.0
        executed = False
        
        if action_type == 'hold':
            executed = True
            
        elif action_type == 'delay':
            # Action delayed due to high gas prices
            executed = False
            
        elif action_type in ['rebalance', 'enter', 'exit']:
            # Calculate transaction costs
            gas_cost = self._calculate_gas_cost(market_data['gas_price'])
            fees = self._calculate_fees(market_data)
            
            # Execute if profitable after costs
            if self._is_profitable_after_costs(action, gas_cost, fees):
                self._update_position(action, market_data)
                self.current_capital -= (gas_cost + fees)
                executed = True
            else:
                executed = False
                action_type = 'unprofitable'
        
        # Record transaction
        transaction = {
            'timestamp': market_data.get('timestamp'),
            'action': action_type,
            'price': market_data['price'],
            'gas_price': market_data['gas_price'],
            'gas_cost': gas_cost,
            'fees': fees,
            'capital': self.current_capital,
            'executed': executed,
            'reason': action.get('reason', ''),
            'confidence': action.get('confidence', 0.5)
        }
        
        self.transaction_history.append(transaction)
        
        return {
            'executed': executed,
            'gas_cost': gas_cost,
            'fees': fees,
            'new_capital': self.current_capital
        }
    
    def _calculate_gas_cost(self, gas_price: float) -> float:
        """Calculate gas cost for transaction"""
        # Approximate gas usage for Uniswap V3 operations
        gas_units = 150000  # Typical for mint/burn operations
        gas_cost_eth = (gas_price * gas_units) / 1e9  # Convert gwei to ETH
        gas_cost_usd = gas_cost_eth * self.current_price  # Convert to USD
        return gas_cost_usd / 10000  # Scale for simulation
    
    def _calculate_fees(self, market_data: Dict[str, Any]) -> float:
        """Calculate trading fees"""
        # Uniswap V3 LP fees are earned, not paid
        # But we simulate opportunity cost and slippage
        return 0.0005  # 0.05% slippage/opportunity cost
    
    def _is_profitable_after_costs(self, action: Dict[str, Any], gas_cost: float, fees: float) -> bool:
        """Check if action is profitable after transaction costs"""
        total_cost = gas_cost + fees
        # Simple profitability check - can be overridden by subclasses
        return total_cost < 0.01  # Less than 1% of capital
    
    def _update_position(self, action: Dict[str, Any], market_data: Dict[str, Any]):
        """Update position state after action execution"""
        action_type = action['action']
        
        if action_type == 'enter':
            self.in_position = True
            self.position_range = action.get('new_range', (0.9, 1.1))  # Default ±10%
            
        elif action_type == 'exit':
            self.in_position = False
            self.position_range = None
            
        elif action_type == 'rebalance':
            self.position_range = action.get('new_range', self.position_range)
        
        # Record position
        position_record = {
            'timestamp': market_data.get('timestamp'),
            'in_position': self.in_position,
            'range': self.position_range,
            'price': market_data['price']
        }
        self.position_history.append(position_record)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        if not self.transaction_history:
            return {}
        
        # Basic metrics
        total_transactions = len(self.transaction_history)
        executed_transactions = sum(1 for t in self.transaction_history if t['executed'])
        total_gas_cost = sum(t['gas_cost'] for t in self.transaction_history)
        total_fees = sum(t['fees'] for t in self.transaction_history)
        
        # ROI calculation
        roi = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Gas efficiency
        gas_per_transaction = total_gas_cost / max(executed_transactions, 1)
        
        # Time in position
        time_in_position = sum(1 for p in self.position_history if p['in_position'])
        position_ratio = time_in_position / max(len(self.position_history), 1)
        
        metrics = {
            'roi': roi,
            'total_transactions': total_transactions,
            'executed_transactions': executed_transactions,
            'execution_rate': executed_transactions / max(total_transactions, 1),
            'total_gas_cost': total_gas_cost,
            'total_fees': total_fees,
            'gas_per_transaction': gas_per_transaction,
            'time_in_position_ratio': position_ratio,
            'final_capital': self.current_capital,
            'net_profit': self.current_capital - self.initial_capital
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def get_recent_volatility(self, window: int = 24) -> float:
        """Calculate recent price volatility"""
        if len(self.price_history) < window:
            return 0.05  # Default 5% volatility
        
        recent_prices = self.price_history[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns) * np.sqrt(24)  # Annualized volatility
    
    def get_gas_percentile(self, window: int = 24, percentile: int = 80) -> float:
        """Get gas price percentile over recent window"""
        if len(self.gas_history) < window:
            return 50.0  # Default gas price
        
        recent_gas = self.gas_history[-window:]
        return np.percentile(recent_gas, percentile)
    
    def is_price_in_range(self, price: Optional[float] = None) -> bool:
        """Check if current price is within position range"""
        if not self.in_position or not self.position_range:
            return False
        
        if price is None:
            price = self.current_price
        
        lower_bound = self.current_price * self.position_range[0]
        upper_bound = self.current_price * self.position_range[1]
        
        return lower_bound <= price <= upper_bound
    
    def reset(self):
        """Reset agent state for new simulation"""
        self.current_capital = self.initial_capital
        self.position_history = []
        self.transaction_history = []
        self.performance_metrics = {}
        self.price_history = []
        self.gas_history = []
        self.volume_history = []
        self.in_position = False
        self.position_range = None
        
        print("Reset agent: {}".format(self.agent_id)) 