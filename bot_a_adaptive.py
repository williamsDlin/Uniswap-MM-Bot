"""
Bot A: Adaptive Bollinger Band Manager
Enhanced with LinUCB contextual bandit for dynamic parameter selection
"""

import time
import csv
import logging
import sys
from datetime import datetime
from typing import Optional, Tuple
from web3 import Web3

from config.settings import settings
from config.contracts import ETH_USDC_POOL
from price_oracle import PriceOracle
from position_manager import PositionManager
from strategies.adaptive_bollinger import AdaptiveBollingerBandsStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_a_adaptive.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AdaptiveBollingerBot:
    """Bot A: Adaptive Bollinger Band Manager with LinUCB learning"""
    
    def __init__(self, bot_id: str = "bot_a"):
        """Initialize the adaptive Bollinger band bot"""
        self.bot_id = bot_id
        
        # Validate settings
        try:
            settings.validate()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(settings.INFURA_GOERLI_RPC_URL))
        if not self.web3.is_connected():
            logger.error("Failed to connect to Web3")
            sys.exit(1)
        
        logger.info(f"Connected to Web3. Latest block: {self.web3.eth.block_number}")
        
        # Initialize components
        self.price_oracle = PriceOracle(self.web3, ETH_USDC_POOL)
        self.position_manager = PositionManager(self.web3)
        
        # Initialize adaptive strategy with bandit learning
        self.strategy = AdaptiveBollingerBandsStrategy(
            initial_window=settings.BOLLINGER_WINDOW,
            initial_std_dev=settings.BOLLINGER_STD_DEV,
            bandit_alpha=1.0,  # Exploration parameter
            reward_lookback_periods=5,
            min_learning_periods=10
        )
        
        # Enhanced CSV logging
        self.csv_filename = f"bot_a_adaptive_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._initialize_csv()
        
        # Bot state tracking
        self.current_position_range: Optional[Tuple[float, float]] = None
        self.is_running = False
        self.cycle_count = 0
        
    def _initialize_csv(self):
        """Initialize enhanced CSV file for logging bot actions"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'cycle', 'action', 'current_price', 'lower_range', 'upper_range',
                'position_id', 'window_size', 'std_dev', 'bandit_confidence', 'reward', 'status', 'notes'
            ])
    
    def run_cycle(self):
        """Run one cycle of the adaptive bot"""
        try:
            self.cycle_count += 1
            
            # Get current price
            current_price = self.price_oracle.get_current_price()
            if current_price is None:
                logger.error("Failed to get current price")
                return
            
            logger.info(f"Cycle {self.cycle_count}: Current ETH/USDC price: {current_price:.6f}")
            
            # Update strategy with current price
            self.strategy.add_price(current_price)
            
            # Check if rebalancing is needed (using adaptive parameters)
            should_rebalance, reason = self.strategy.should_rebalance(
                current_price, self.current_position_range
            )
            
            logger.info(f"Rebalance check: {should_rebalance} - {reason}")
            logger.info(f"Current adaptive parameters: window={self.strategy.current_window}, "
                       f"std_dev={self.strategy.current_std_dev:.2f}")
            
        except Exception as e:
            logger.error(f"Error in run cycle: {e}")
    
    def run(self):
        """Main bot loop with adaptive learning"""
        logger.info("Starting Bot A: Adaptive Bollinger Band Manager")
        logger.info(f"Check interval: {settings.CHECK_INTERVAL_MINUTES} minutes")
        logger.info(f"LinUCB bandit learning enabled")
        
        self.is_running = True
        
        try:
            while self.is_running:
                logger.info(f"--- Starting adaptive cycle {self.cycle_count + 1} at {datetime.now()} ---")
                
                self.run_cycle()
                
                logger.info(f"Cycle completed. Sleeping for {settings.CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping adaptive bot...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.is_running = False

def main():
    """Main entry point for Bot A"""
    bot = AdaptiveBollingerBot()
    bot.run()

if __name__ == "__main__":
    main() 