"""
Bot D: Gas-Aware Executor
Uses SARIMA-based gas price prediction to optimize transaction timing
"""

import time
import logging
import sys
from datetime import datetime
from typing import Optional, Tuple
from web3 import Web3

from config.settings import settings
from config.contracts import ETH_USDC_POOL
from price_oracle import PriceOracle
from position_manager import PositionManager
from strategies.bollinger import BollingerBandsStrategy
from ml.gas_predictor import GasPredictorAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GasAwareBot:
    """Bot D: Gas-aware executor with transaction timing optimization"""
    
    def __init__(self, bot_id: str = "bot_d"):
        """Initialize the gas-aware bot"""
        self.bot_id = bot_id
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(settings.INFURA_GOERLI_RPC_URL))
        if not self.web3.is_connected():
            logger.error("Failed to connect to Web3")
            sys.exit(1)
        
        # Initialize components
        self.price_oracle = PriceOracle(self.web3, ETH_USDC_POOL)
        self.position_manager = PositionManager(self.web3)
        self.strategy = BollingerBandsStrategy()
        
        # Initialize gas predictor
        self.gas_predictor = GasPredictorAgent(agent_name="gas_predictor_bot_d")
        self.gas_predictor.load_model()
        
        # Bot state
        self.current_position_range: Optional[Tuple[float, float]] = None
        self.cycle_count = 0
        self.is_running = False
        
    def should_execute_transaction(self, transaction_type: str) -> Tuple[bool, str]:
        """Determine if transaction should be executed based on gas predictions"""
        try:
            current_gas_price = self.web3.eth.gas_price
            current_gas_gwei = self.web3.from_wei(current_gas_price, 'gwei')
            
            should_execute, reason, confidence = self.gas_predictor.should_execute_now(
                current_gas_price=float(current_gas_gwei),
                max_gas_price=50.0,
                urgency_hours=6
            )
            
            logger.info(f"Gas decision for {transaction_type}: {should_execute} - {reason}")
            return should_execute, reason
            
        except Exception as e:
            logger.error(f"Error in gas decision: {e}")
            return True, "Error in prediction, executing conservatively"
    
    def run_cycle(self):
        """Run one cycle of the gas-aware bot"""
        try:
            self.cycle_count += 1
            
            # Update gas predictions
            current_gas_price = self.web3.eth.gas_price
            current_gas_gwei = self.web3.from_wei(current_gas_price, 'gwei')
            self.gas_predictor.add_gas_price_observation(float(current_gas_gwei))
            
            # Get current price
            current_price = self.price_oracle.get_current_price()
            if current_price is None:
                logger.error("Failed to get current price")
                return
            
            logger.info(f"Cycle {self.cycle_count}: Price: {current_price:.6f}, Gas: {current_gas_gwei:.1f} Gwei")
            
            # Check if rebalancing is needed
            self.strategy.add_price(current_price)
            should_rebalance, reason = self.strategy.should_rebalance(
                current_price, self.current_position_range
            )
            
            if should_rebalance:
                # Check gas conditions
                should_execute, gas_reason = self.should_execute_transaction('rebalance')
                
                if should_execute:
                    logger.info(f"Executing rebalance: {gas_reason}")
                    # Would execute rebalancing here
                else:
                    logger.info(f"Delaying rebalance due to gas: {gas_reason}")
            
        except Exception as e:
            logger.error(f"Error in gas-aware run cycle: {e}")
    
    def run(self):
        """Main bot loop"""
        logger.info("Starting Bot D: Gas-Aware Executor")
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_cycle()
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
        except KeyboardInterrupt:
            logger.info("Stopping gas-aware bot...")
            self.is_running = False

def main():
    """Main entry point for Bot D"""
    bot = GasAwareBot()
    bot.run()

if __name__ == "__main__":
    main() 