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
        self.last_position_value = 0.0
        self.last_fees_earned = 0.0
        self.is_running = False
        self.cycle_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_rebalances': 0,
            'total_fees_earned': 0.0,
            'total_gas_spent': 0.0,
            'parameter_adaptations': 0,
            'successful_positions': 0,
            'failed_positions': 0
        }
        
    def _initialize_csv(self):
        """Initialize enhanced CSV file for logging bot actions"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'cycle', 'action', 'current_price', 'lower_range', 'upper_range',
                'position_id', 'eth_amount', 'usdc_amount', 'gas_used', 'tx_hash', 'status', 
                'window_size', 'std_dev', 'bandit_confidence', 'reward', 'context_features', 'notes'
            ])
    
    def log_action(self, action: str, current_price: float, **kwargs):
        """Enhanced action logging with bandit information"""
        try:
            # Get current strategy parameters
            window_size = getattr(self.strategy, 'current_window', 'N/A')
            std_dev = getattr(self.strategy, 'current_std_dev', 'N/A')
            
            # Get bandit stats
            bandit_stats = self.strategy.bandit.get_stats()
            confidence = kwargs.get('bandit_confidence', 'N/A')
            reward = kwargs.get('reward', 'N/A')
            context = kwargs.get('context_features', 'N/A')
            
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.cycle_count,
                    action,
                    current_price,
                    kwargs.get('lower_range', ''),
                    kwargs.get('upper_range', ''),
                    kwargs.get('position_id', ''),
                    kwargs.get('eth_amount', ''),
                    kwargs.get('usdc_amount', ''),
                    kwargs.get('gas_used', ''),
                    kwargs.get('tx_hash', ''),
                    kwargs.get('status', ''),
                    window_size,
                    std_dev,
                    confidence,
                    reward,
                    context,
                    kwargs.get('notes', '')
                ])
        except Exception as e:
            logger.error(f"Failed to log action to CSV: {e}")
    
    def check_safety_conditions(self) -> bool:
        """Check safety conditions before executing trades"""
        try:
            # Check ETH balance
            eth_balance, _ = self.position_manager.get_token_balances()
            if eth_balance < settings.MIN_ETH_BALANCE:
                logger.warning(f"ETH balance too low: {eth_balance:.6f} < {settings.MIN_ETH_BALANCE}")
                return False
            
            # Check gas price
            gas_price = self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price, 'gwei')
            if gas_price_gwei > settings.MAX_GAS_PRICE_GWEI:
                logger.warning(f"Gas price too high: {gas_price_gwei:.2f} > {settings.MAX_GAS_PRICE_GWEI}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking safety conditions: {e}")
            return False
    
    def estimate_position_value(self) -> Tuple[float, float]:
        """
        Estimate current position value and fees earned
        
        Returns:
            Tuple of (position_value, fees_earned_since_last_check)
        """
        try:
            if not self.position_manager.current_position_id:
                return 0.0, 0.0
            
            # Get position info
            position_info = self.position_manager.get_position_info(
                self.position_manager.current_position_id
            )
            
            if not position_info:
                return 0.0, 0.0
            
            # Simple value estimation (could be enhanced with actual calculations)
            # This is a placeholder - in practice you'd calculate based on:
            # - Current liquidity amount
            # - Current price vs position range
            # - Accumulated fees
            
            current_price = self.price_oracle.get_current_price()
            if current_price is None:
                return self.last_position_value, 0.0
            
            # Rough position value estimation
            liquidity = position_info.get('liquidity', 0)
            if liquidity > 0:
                # This is a simplified calculation
                estimated_value = liquidity * current_price * 0.0001  # Rough conversion
            else:
                estimated_value = 0.0
            
            # Estimate fees earned (placeholder calculation)
            fees_earned = max(0.0, estimated_value - self.last_position_value) * 0.003  # 0.3% fee estimate
            
            return estimated_value, fees_earned
            
        except Exception as e:
            logger.error(f"Error estimating position value: {e}")
            return self.last_position_value, 0.0
    
    def rebalance_position(self, current_price: float) -> bool:
        """Enhanced rebalance with bandit feedback"""
        try:
            # Get current position value for reward calculation
            position_value, fees_earned = self.estimate_position_value()
            
            # Get optimal range from adaptive strategy
            lower_price, upper_price = self.strategy.get_optimal_range(current_price)
            
            # Get current balances
            eth_balance, usdc_balance = self.position_manager.get_token_balances()
            
            # Calculate position size
            eth_amount, usdc_amount = self.strategy.get_position_size(eth_balance, current_price)
            
            gas_used = 0
            
            # Remove existing position if any
            if self.position_manager.current_position_id:
                logger.info(f"Removing existing position: {self.position_manager.current_position_id}")
                
                success = self.position_manager.remove_liquidity(
                    self.position_manager.current_position_id,
                    100.0
                )
                
                if not success:
                    self.log_action(
                        'remove_liquidity_failed',
                        current_price,
                        position_id=self.position_manager.current_position_id,
                        status='failed',
                        notes='Failed to remove existing liquidity'
                    )
                    self.performance_metrics['failed_positions'] += 1
                    return False
                
                # Record rebalance event in strategy
                self.strategy.record_rebalance_event(
                    old_range=self.current_position_range,
                    new_range=(lower_price, upper_price),
                    gas_cost=0.001,  # Estimated gas cost
                    reason="Adaptive rebalancing"
                )
                
                gas_used += 150000  # Estimated gas for removal
            
            # Convert price range to ticks
            lower_tick, upper_tick = self.price_oracle.get_tick_range_for_price_range(
                lower_price, upper_price
            )
            
            # Approve tokens
            logger.info(f"Approving tokens: {eth_amount:.6f} ETH, {usdc_amount:.2f} USDC")
            if not self.position_manager.approve_tokens(eth_amount, usdc_amount):
                self.log_action(
                    'approve_failed',
                    current_price,
                    eth_amount=eth_amount,
                    usdc_amount=usdc_amount,
                    status='failed'
                )
                return False
            
            gas_used += 100000  # Estimated gas for approvals
            
            # Mint new position
            logger.info(f"Minting position with adaptive range [{lower_price:.6f}, {upper_price:.6f}] "
                       f"(window={self.strategy.current_window}, std={self.strategy.current_std_dev:.2f})")
            
            position_id = self.position_manager.mint_position(
                lower_tick,
                upper_tick,
                eth_amount,
                usdc_amount
            )
            
            gas_used += 400000  # Estimated gas for minting
            
            if position_id:
                self.current_position_range = (lower_price, upper_price)
                self.last_position_value = position_value
                self.last_fees_earned += fees_earned
                
                self.performance_metrics['total_rebalances'] += 1
                self.performance_metrics['total_gas_spent'] += gas_used * 30e-9  # Rough gas cost in ETH
                self.performance_metrics['successful_positions'] += 1
                
                self.log_action(
                    'adaptive_rebalance_success',
                    current_price,
                    lower_range=lower_price,
                    upper_range=upper_price,
                    position_id=position_id,
                    eth_amount=eth_amount,
                    usdc_amount=usdc_amount,
                    gas_used=gas_used,
                    status='success',
                    reward=fees_earned,
                    notes=f"Adaptive parameters: window={self.strategy.current_window}, std={self.strategy.current_std_dev:.2f}"
                )
                
                logger.info(f"Successfully rebalanced with adaptive parameters. New position ID: {position_id}")
                return True
            else:
                self.performance_metrics['failed_positions'] += 1
                self.log_action(
                    'mint_position_failed',
                    current_price,
                    lower_range=lower_price,
                    upper_range=upper_price,
                    eth_amount=eth_amount,
                    usdc_amount=usdc_amount,
                    status='failed'
                )
                return False
                
        except Exception as e:
            logger.error(f"Error in adaptive rebalancing: {e}")
            self.log_action(
                'rebalance_error',
                current_price,
                status='error',
                notes=str(e)
            )
            return False
    
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
            
            # Get position value and fees for strategy feedback
            position_value, fees_earned = self.estimate_position_value()
            
            # Update strategy with current price and performance data
            self.strategy.add_price(
                current_price,
                position_value=position_value,
                fees_earned=fees_earned,
                gas_cost=0.001  # Estimated gas cost per cycle
            )
            
            # Check if rebalancing is needed (using adaptive parameters)
            should_rebalance, reason = self.strategy.should_rebalance(
                current_price, self.current_position_range
            )
            
            logger.info(f"Rebalance check: {should_rebalance} - {reason}")
            logger.info(f"Current adaptive parameters: window={self.strategy.current_window}, "
                       f"std_dev={self.strategy.current_std_dev:.2f}")
            
            if should_rebalance:
                # Check safety conditions
                if not self.check_safety_conditions():
                    logger.warning("Safety conditions not met, skipping rebalance")
                    self.log_action(
                        'safety_check_failed',
                        current_price,
                        status='skipped',
                        notes='Safety conditions not met'
                    )
                    return
                
                # Perform adaptive rebalancing
                logger.info("Performing adaptive rebalance...")
                success = self.rebalance_position(current_price)
                
                if success:
                    logger.info("Adaptive rebalance completed successfully")
                else:
                    logger.error("Adaptive rebalance failed")
            else:
                # Log price update with current strategy state
                self.log_action(
                    'price_update',
                    current_price,
                    status='no_action',
                    notes=f"No rebalance needed: {reason}"
                )
            
            # Log strategy statistics
            stats = self.strategy.get_enhanced_statistics()
            logger.debug(f"Enhanced strategy stats: {stats}")
            
            # Update performance metrics
            self.performance_metrics['total_fees_earned'] += fees_earned
            
            # Save model periodically
            if self.cycle_count % 10 == 0:
                self.strategy.save_model()
                logger.info("Saved adaptive strategy model")
            
        except Exception as e:
            logger.error(f"Error in run cycle: {e}")
    
    def run(self):
        """Main bot loop with adaptive learning"""
        logger.info("Starting Bot A: Adaptive Bollinger Band Manager")
        logger.info(f"Check interval: {settings.CHECK_INTERVAL_MINUTES} minutes")
        logger.info(f"Initial Bollinger parameters: {settings.BOLLINGER_WINDOW} periods, {settings.BOLLINGER_STD_DEV} std dev")
        logger.info(f"LinUCB bandit learning enabled with alpha=1.0")
        
        # Log pool info
        pool_info = self.price_oracle.get_pool_info()
        logger.info(f"Pool info: {pool_info}")
        
        self.is_running = True
        
        try:
            while self.is_running:
                logger.info(f"--- Starting adaptive cycle {self.cycle_count + 1} at {datetime.now()} ---")
                
                self.run_cycle()
                
                # Log performance summary
                logger.info(f"Performance summary: {self.performance_metrics}")
                
                logger.info(f"Cycle completed. Sleeping for {settings.CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping adaptive bot...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.is_running = False
        finally:
            # Save final model state
            self.strategy.save_model()
            logger.info("Final model state saved")

def main():
    """Main entry point for Bot A"""
    bot = AdaptiveBollingerBot()
    bot.run()

if __name__ == "__main__":
    main() 