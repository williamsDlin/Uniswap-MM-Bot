"""
Bot B: Reinforcement Learning Rebalancer
Uses PPO agent to make intelligent rebalancing decisions
"""

import time
import csv
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple
from web3 import Web3

from config.settings import settings
from config.contracts import ETH_USDC_POOL
from price_oracle import PriceOracle
from position_manager import PositionManager
from strategies.bollinger import BollingerBandsStrategy
from ml.rebalance_rl_agent import RebalanceRLAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_b_rl.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class RLRebalancerBot:
    """Bot B: RL-driven intelligent rebalancer"""
    
    def __init__(self, bot_id: str = "bot_b"):
        """Initialize the RL rebalancer bot"""
        self.bot_id = bot_id
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(settings.INFURA_GOERLI_RPC_URL))
        if not self.web3.is_connected():
            logger.error("Failed to connect to Web3")
            sys.exit(1)
        
        # Initialize components
        self.price_oracle = PriceOracle(self.web3, ETH_USDC_POOL)
        self.position_manager = PositionManager(self.web3)
        
        # Initialize strategy for range calculations
        self.strategy = BollingerBandsStrategy()
        
        # Initialize RL agent
        self.rl_agent = RebalanceRLAgent(agent_name="rebalance_ppo_agent")
        self.rl_agent.load_model()
        
        # Bot state
        self.current_position_range: Optional[Tuple[float, float]] = None
        self.cycle_count = 0
        self.is_running = False
        
    def run_cycle(self):
        """Run one cycle of the RL bot"""
        try:
            self.cycle_count += 1
            
            # Get current price
            current_price = self.price_oracle.get_current_price()
            if current_price is None:
                logger.error("Failed to get current price")
                return
            
            logger.info(f"Cycle {self.cycle_count}: Current ETH/USDC price: {current_price:.6f}")
            
            # Extract state for RL agent
            gas_price = self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price, 'gwei')
            
            state = self.rl_agent.extract_state(
                current_price=current_price,
                position_range=self.current_position_range,
                fees_earned=0.001,
                gas_price_gwei=float(gas_price_gwei),
                time_since_rebalance=1.0,
                price_history=[current_price] * 10
            )
            
            # Get action from RL agent
            action = self.rl_agent.select_action(state)
            action_name = self.rl_agent.get_action_name(action)
            
            logger.info(f"RL Agent decision: {action_name}")
            
        except Exception as e:
            logger.error(f"Error in RL run cycle: {e}")
    
    def run(self):
        """Main bot loop"""
        logger.info("Starting Bot B: RL-driven Rebalancer")
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
        except KeyboardInterrupt:
            logger.info("Stopping RL bot...")
            self.is_running = False

def main():
    """Main entry point for Bot B"""
    bot = RLRebalancerBot()
    bot.run()

if __name__ == "__main__":
    main() 