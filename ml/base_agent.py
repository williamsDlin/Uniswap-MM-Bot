"""
Base class for ML agents in the Uniswap V3 bot framework
"""

import os
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all ML agents"""
    
    def __init__(self, agent_name: str, model_dir: str = "models"):
        """
        Initialize base agent
        
        Args:
            agent_name: Unique identifier for this agent
            model_dir: Directory to save/load models
        """
        self.agent_name = agent_name
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, f"{agent_name}.pkl")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.decision_history = []
        
        logger.info(f"Initialized {agent_name} agent")
    
    @abstractmethod
    def select_action(self, context: np.ndarray) -> Any:
        """
        Select action based on context
        
        Args:
            context: Context features
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, context: np.ndarray, action: Any, reward: float) -> None:
        """
        Update agent based on observed reward
        
        Args:
            context: Context features
            action: Action taken
            reward: Observed reward
        """
        pass
    
    def save_model(self) -> None:
        """Save model state to disk"""
        try:
            model_state = {
                'agent_name': self.agent_name,
                'performance_history': self.performance_history,
                'decision_history': self.decision_history,
                'timestamp': datetime.now().isoformat(),
                'model_data': self._get_model_state()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_state, f)
            
            logger.info(f"Saved {self.agent_name} model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self) -> bool:
        """
        Load model state from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"No saved model found for {self.agent_name}")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_state = pickle.load(f)
            
            self.performance_history = model_state.get('performance_history', [])
            self.decision_history = model_state.get('decision_history', [])
            
            self._set_model_state(model_state.get('model_data', {}))
            
            logger.info(f"Loaded {self.agent_name} model from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for serialization"""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model-specific state from loaded data"""
        pass
    
    def log_decision(self, context: np.ndarray, action: Any, confidence: float = None) -> None:
        """Log decision for analysis"""
        decision = {
            'timestamp': datetime.now().isoformat(),
            'context': context.tolist() if isinstance(context, np.ndarray) else context,
            'action': action,
            'confidence': confidence
        }
        self.decision_history.append(decision)
        
        # Keep only recent decisions (last 1000)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def log_performance(self, reward: float, additional_metrics: Dict[str, float] = None) -> None:
        """Log performance metrics"""
        performance = {
            'timestamp': datetime.now().isoformat(),
            'reward': reward,
            'cumulative_reward': sum([p['reward'] for p in self.performance_history]) + reward
        }
        
        if additional_metrics:
            performance.update(additional_metrics)
        
        self.performance_history.append(performance)
        
        # Keep only recent performance (last 1000)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if not self.performance_history:
            return {"status": "no_data"}
        
        rewards = [p['reward'] for p in self.performance_history]
        
        return {
            'agent_name': self.agent_name,
            'total_decisions': len(self.decision_history),
            'total_updates': len(self.performance_history),
            'average_reward': np.mean(rewards),
            'total_reward': sum(rewards),
            'reward_std': np.std(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'recent_performance': rewards[-10:] if len(rewards) >= 10 else rewards
        } 