"""
LinUCB Contextual Bandit for adaptive Bollinger Band parameter selection
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Any
from ml.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class LinUCBBanditAgent(BaseAgent):
    """
    LinUCB Contextual Bandit for selecting optimal Bollinger Band parameters
    
    Actions: (window_size, std_dev) combinations
    Context: Market features (volatility, trend, momentum, etc.)
    Reward: Based on position performance (fees earned - rebalance costs)
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 context_dim: int = 8,
                 agent_name: str = "bollinger_bandit"):
        """
        Initialize LinUCB bandit
        
        Args:
            alpha: Exploration parameter (higher = more exploration)
            context_dim: Dimension of context features
            agent_name: Name for this agent
        """
        super().__init__(agent_name)
        
        self.alpha = alpha
        self.context_dim = context_dim
        
        # Define action space: (window_size, std_dev) combinations
        self.actions = self._create_action_space()
        self.n_actions = len(self.actions)
        
        # LinUCB parameters for each action
        self.A = {}  # Ridge regression matrices
        self.b = {}  # Right-hand side vectors
        
        # Initialize matrices for each action
        for i, action in enumerate(self.actions):
            self.A[i] = np.identity(context_dim)
            self.b[i] = np.zeros(context_dim)
        
        # Performance tracking
        self.action_counts = np.zeros(self.n_actions)
        self.recent_rewards = []
        
        logger.info(f"LinUCB agent initialized with {self.n_actions} actions, context_dim={context_dim}")
    
    def _create_action_space(self) -> List[Tuple[int, float]]:
        """Create action space of (window_size, std_dev) combinations"""
        window_sizes = [10, 15, 20, 25, 30, 40, 50]
        std_devs = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        actions = []
        for window in window_sizes:
            for std_dev in std_devs:
                actions.append((window, std_dev))
        
        return actions
    
    def extract_context(self, 
                       price_history: List[float],
                       volatility_window: int = 20) -> np.ndarray:
        """
        Extract context features from market data
        
        Args:
            price_history: Recent price history
            volatility_window: Window for volatility calculation
            
        Returns:
            Context feature vector
        """
        if len(price_history) < volatility_window:
            # Return default context if insufficient data
            return np.ones(self.context_dim) * 0.5
        
        prices = np.array(price_history[-volatility_window:])
        returns = np.diff(prices) / prices[:-1]
        
        # Feature 1: Short-term volatility (last 5 periods)
        short_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0.1
        
        # Feature 2: Long-term volatility (full window)
        long_vol = np.std(returns)
        
        # Feature 3: Volatility ratio (short/long)
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        
        # Feature 4: Price momentum (trend strength)
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # Feature 5: Mean reversion indicator
        current_price = prices[-1]
        mean_price = np.mean(prices)
        mean_reversion = (current_price - mean_price) / mean_price
        
        # Feature 6: Range expansion/contraction
        recent_range = np.max(prices[-5:]) - np.min(prices[-5:])
        avg_range = np.mean([np.max(prices[i:i+5]) - np.min(prices[i:i+5]) 
                            for i in range(len(prices)-5)])
        range_ratio = recent_range / avg_range if avg_range > 0 else 1.0
        
        # Feature 7: Price level (normalized)
        price_level = current_price / mean_price
        
        # Feature 8: Time-based feature (hour of day normalized)
        from datetime import datetime
        hour = datetime.now().hour / 24.0
        
        context = np.array([
            min(short_vol * 100, 10.0),  # Scale and cap volatility
            min(long_vol * 100, 10.0),
            min(vol_ratio, 5.0),
            np.tanh(momentum * 10),  # Bound momentum
            np.tanh(mean_reversion * 5),  # Bound mean reversion
            min(range_ratio, 3.0),
            min(price_level, 2.0),
            hour
        ])
        
        # Normalize to [0, 1] range
        context = (context - context.min()) / (context.max() - context.min() + 1e-8)
        
        return context
    
    def select_action(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Select action using LinUCB algorithm
        
        Args:
            context: Context feature vector
            
        Returns:
            (window_size, std_dev) tuple
        """
        if context.shape[0] != self.context_dim:
            logger.warning(f"Context dimension mismatch: {context.shape[0]} != {self.context_dim}")
            context = np.resize(context, self.context_dim)
        
        ucb_values = np.zeros(self.n_actions)
        
        for i in range(self.n_actions):
            # Compute ridge regression coefficients
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            
            # Compute confidence interval
            confidence_width = self.alpha * np.sqrt(context.T @ A_inv @ context)
            
            # UCB value = expected reward + confidence interval
            ucb_values[i] = theta.T @ context + confidence_width
        
        # Select action with highest UCB value
        selected_action_idx = np.argmax(ucb_values)
        selected_action = self.actions[selected_action_idx]
        
        # Update action count
        self.action_counts[selected_action_idx] += 1
        
        # Log decision
        confidence = ucb_values[selected_action_idx] - np.mean(ucb_values)
        self.log_decision(context, selected_action, confidence)
        
        logger.debug(f"Selected action {selected_action} with UCB value {ucb_values[selected_action_idx]:.4f}")
        
        return selected_action
    
    def update(self, context: np.ndarray, action: Tuple[int, float], reward: float) -> None:
        """
        Update LinUCB parameters based on observed reward
        
        Args:
            context: Context features when action was taken
            action: Action that was taken
            reward: Observed reward
        """
        # Find action index
        action_idx = None
        for i, a in enumerate(self.actions):
            if a == action:
                action_idx = i
                break
        
        if action_idx is None:
            logger.error(f"Unknown action: {action}")
            return
        
        if context.shape[0] != self.context_dim:
            context = np.resize(context, self.context_dim)
        
        # Update LinUCB parameters
        self.A[action_idx] += np.outer(context, context)
        self.b[action_idx] += reward * context
        
        # Track performance
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        self.log_performance(reward, {
            'action_window': action[0],
            'action_std_dev': action[1],
            'action_count': self.action_counts[action_idx],
            'avg_recent_reward': np.mean(self.recent_rewards)
        })
        
        logger.debug(f"Updated parameters for action {action} with reward {reward:.4f}")
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about action selection"""
        total_actions = np.sum(self.action_counts)
        
        if total_actions == 0:
            return {"status": "no_actions"}
        
        action_probs = self.action_counts / total_actions
        
        # Find most and least used actions
        most_used_idx = np.argmax(self.action_counts)
        least_used_idx = np.argmin(self.action_counts)
        
        return {
            'total_actions': int(total_actions),
            'action_distribution': {
                f"window_{self.actions[i][0]}_std_{self.actions[i][1]}": {
                    'count': int(self.action_counts[i]),
                    'probability': float(action_probs[i])
                }
                for i in range(min(10, self.n_actions))  # Top 10 actions
            },
            'most_used_action': {
                'action': self.actions[most_used_idx],
                'count': int(self.action_counts[most_used_idx]),
                'probability': float(action_probs[most_used_idx])
            },
            'least_used_action': {
                'action': self.actions[least_used_idx],
                'count': int(self.action_counts[least_used_idx]),
                'probability': float(action_probs[least_used_idx])
            },
            'exploration_entropy': float(-np.sum(action_probs * np.log(action_probs + 1e-10)))
        }
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization"""
        return {
            'alpha': self.alpha,
            'context_dim': self.context_dim,
            'actions': self.actions,
            'A': {k: v.tolist() for k, v in self.A.items()},
            'b': {k: v.tolist() for k, v in self.b.items()},
            'action_counts': self.action_counts.tolist(),
            'recent_rewards': self.recent_rewards
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loaded data"""
        self.alpha = state.get('alpha', self.alpha)
        self.context_dim = state.get('context_dim', self.context_dim)
        self.actions = state.get('actions', self.actions)
        self.action_counts = np.array(state.get('action_counts', np.zeros(self.n_actions)))
        self.recent_rewards = state.get('recent_rewards', [])
        
        # Restore matrices
        A_data = state.get('A', {})
        b_data = state.get('b', {})
        
        for i in range(self.n_actions):
            if str(i) in A_data:
                self.A[i] = np.array(A_data[str(i)])
            if str(i) in b_data:
                self.b[i] = np.array(b_data[str(i)]) 