"""
PPO-based Reinforcement Learning Agent for Rebalancing Decisions
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from ml.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPONetwork, self).__init__()
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.shared_layers(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class RebalanceRLAgent(BaseAgent):
    """
    PPO-based RL agent for making rebalancing decisions
    
    State: [price_deviation, range_width, fees_earned, gas_price, time_since_last_rebalance, volatility, momentum, position_health]
    Actions: [NO_ACTION, REBALANCE_NARROW, REBALANCE_WIDE, REBALANCE_SHIFT_UP, REBALANCE_SHIFT_DOWN]
    """
    
    def __init__(self, 
                 state_dim: int = 8,
                 action_dim: int = 5,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 agent_name: str = "rebalance_rl_agent"):
        """
        Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs for PPO update
            agent_name: Name for this agent
        """
        super().__init__(agent_name)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        # Action mapping
        self.actions = [
            'NO_ACTION',           # 0: Do nothing
            'REBALANCE_NARROW',    # 1: Narrow the range
            'REBALANCE_WIDE',      # 2: Widen the range  
            'REBALANCE_SHIFT_UP',  # 3: Shift range up
            'REBALANCE_SHIFT_DOWN' # 4: Shift range down
        ]
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
        logger.info(f"PPO RL agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def extract_state(self, 
                     current_price: float,
                     position_range: Optional[Tuple[float, float]],
                     fees_earned: float,
                     gas_price_gwei: float,
                     time_since_rebalance: float,
                     price_history: list,
                     position_value: float = 1.0) -> np.ndarray:
        """
        Extract state features for RL decision making
        
        Args:
            current_price: Current market price
            position_range: Current position range (lower, upper) or None
            fees_earned: Fees earned since last rebalance
            gas_price_gwei: Current gas price in Gwei
            time_since_rebalance: Hours since last rebalance
            price_history: Recent price history
            position_value: Current position value
            
        Returns:
            State vector
        """
        if position_range is None:
            # No active position
            return np.array([
                0.0,  # price_deviation_ratio
                0.0,  # range_width_ratio
                fees_earned,
                min(gas_price_gwei / 100.0, 1.0),  # Normalized gas price
                min(time_since_rebalance / 24.0, 1.0),  # Normalized time
                0.5,  # volatility (placeholder)
                0.0,  # momentum
                0.0   # position_health
            ])
        
        lower_range, upper_range = position_range
        
        # Feature 1: Price deviation from range center
        range_center = (lower_range + upper_range) / 2
        price_deviation_ratio = (current_price - range_center) / range_center
        
        # Feature 2: Range width relative to price
        range_width = upper_range - lower_range
        range_width_ratio = range_width / current_price
        
        # Feature 3: Fees earned (normalized)
        fees_normalized = min(fees_earned * 1000, 1.0)  # Scale fees
        
        # Feature 4: Gas price (normalized)
        gas_normalized = min(gas_price_gwei / 100.0, 1.0)
        
        # Feature 5: Time since last rebalance (normalized)
        time_normalized = min(time_since_rebalance / 24.0, 1.0)
        
        # Feature 6: Volatility
        if len(price_history) >= 10:
            recent_prices = np.array(price_history[-10:])
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0.1
        volatility_normalized = min(volatility * 50, 1.0)
        
        # Feature 7: Momentum
        if len(price_history) >= 5:
            momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
        else:
            momentum = 0.0
        momentum_normalized = np.tanh(momentum * 10)
        
        # Feature 8: Position health (how well positioned we are)
        if lower_range <= current_price <= upper_range:
            position_health = 1.0 - abs(price_deviation_ratio)  # Closer to center is healthier
        else:
            position_health = 0.0  # Outside range is unhealthy
        
        state = np.array([
            np.tanh(price_deviation_ratio * 5),  # Bound price deviation
            min(range_width_ratio, 1.0),
            fees_normalized,
            gas_normalized,
            time_normalized,
            volatility_normalized,
            momentum_normalized,
            position_health
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using current policy
        
        Args:
            state: State vector
            deterministic: If True, select action deterministically
            
        Returns:
            Selected action index
        """
        if state.shape[0] != self.state_dim:
            state = np.resize(state, self.state_dim).astype(np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
        
        return action
    
    def select_action_with_logprob(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action and return log probability for training"""
        if state.shape[0] != self.state_dim:
            state = np.resize(state, self.state_dim).astype(np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        action_probs, _ = self.policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        return action.item(), logprob.item()
    
    def store_transition(self, state: np.ndarray, action: int, logprob: float, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
        self.memory['dones'].append(done)
    
    def calculate_returns(self, rewards: list, dones: list) -> list:
        """Calculate discounted returns"""
        returns = []
        discounted_sum = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        return returns
    
    def update(self, context: np.ndarray = None, action: Any = None, reward: float = 0.0) -> None:
        """
        Update policy using PPO algorithm
        
        Args:
            context: Not used in PPO (uses stored memory)
            action: Not used in PPO
            reward: Not used in PPO (uses stored rewards)
        """
        if len(self.memory['states']) < 32:  # Minimum batch size
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_logprobs = torch.FloatTensor(self.memory['logprobs']).to(self.device)
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        
        # Calculate returns
        returns = self.calculate_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        total_loss = 0
        
        # PPO update for k epochs
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_probs, state_values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Calculate ratio
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # Calculate advantages
            advantages = returns - state_values.squeeze()
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(state_values.squeeze(), returns)
            
            # Entropy bonus for exploration
            entropy_loss = -0.01 * entropy.mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            total_loss += loss.item()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear memory
        for key in self.memory:
            self.memory[key].clear()
        
        # Log training statistics
        avg_loss = total_loss / self.k_epochs
        self.training_losses.append(avg_loss)
        
        # Log performance
        if rewards:
            episode_reward = sum(rewards)
            self.episode_rewards.append(episode_reward)
            self.log_performance(episode_reward, {
                'policy_loss': avg_loss,
                'episode_length': len(rewards),
                'average_return': returns.mean().item()
            })
        
        logger.debug(f"PPO update completed. Average loss: {avg_loss:.4f}")
    
    def get_action_name(self, action_idx: int) -> str:
        """Get human-readable action name"""
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        return f"UNKNOWN_{action_idx}"
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'eps_clip': self.eps_clip,
            'k_epochs': self.k_epochs,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model state from loaded data"""
        self.state_dim = state.get('state_dim', self.state_dim)
        self.action_dim = state.get('action_dim', self.action_dim)
        self.gamma = state.get('gamma', self.gamma)
        self.eps_clip = state.get('eps_clip', self.eps_clip)
        self.k_epochs = state.get('k_epochs', self.k_epochs)
        
        # Restore model weights
        if 'policy_state_dict' in state:
            self.policy.load_state_dict(state['policy_state_dict'])
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Restore training history
        self.episode_rewards = state.get('episode_rewards', [])
        self.episode_lengths = state.get('episode_lengths', [])
        self.training_losses = state.get('training_losses', []) 