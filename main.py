"""
Uniswap V3 Liquidity Management Bot
Main application loop
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
from strategies.bollinger import BollingerBandsStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class UniswapV3Bot:
    def __init__(self):
        """Initialize the Uniswap V3 liquidity management bot"""
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
        self.strategy = BollingerBandsStrategy(
            window=settings.BOLLINGER_WINDOW,
            std_dev=settings.BOLLINGER_STD_DEV
        )
        
        # Initialize CSV logging
        self.csv_filename = f"bot_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._initialize_csv()
        
        # Bot state
        self.current_position_range: Optional[Tuple[float, float]] = None
        self.is_running = False
        
    def _initialize_csv(self):
        """Initialize CSV file for logging bot actions"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'action', 'current_price', 'lower_range', 'upper_range',
                'position_id', 'eth_amount', 'usdc_amount', 'gas_used', 'tx_hash', 'status', 'notes'
            ])
    
    def log_action(self, action: str, current_price: float, **kwargs):
        """Log action to CSV file"""
        try:
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    datetime.now().isoformat(),
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
    
    def rebalance_position(self, current_price: float) -> bool:
        """Rebalance the current position based on strategy"""
        try:
            # Get optimal range from strategy
            lower_price, upper_price = self.strategy.get_optimal_range(current_price)
            
            # Get current balances
            eth_balance, usdc_balance = self.position_manager.get_token_balances()
            
            # Calculate position size
            eth_amount, usdc_amount = self.strategy.get_position_size(eth_balance, current_price)
            
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
                    return False
                
                self.log_action(
                    'remove_liquidity',
                    current_price,
                    position_id=self.position_manager.current_position_id,
                    status='success'
                )
            
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
            
            # Mint new position
            logger.info(f"Minting position with range [{lower_price:.6f}, {upper_price:.6f}] (ticks: [{lower_tick}, {upper_tick}])")
            
            position_id = self.position_manager.mint_position(
                lower_tick,
                upper_tick,
                eth_amount,
                usdc_amount
            )
            
            if position_id:
                self.current_position_range = (lower_price, upper_price)
                
                self.log_action(
                    'mint_position',
                    current_price,
                    lower_range=lower_price,
                    upper_range=upper_price,
                    position_id=position_id,
                    eth_amount=eth_amount,
                    usdc_amount=usdc_amount,
                    status='success'
                )
                
                logger.info(f"Successfully rebalanced. New position ID: {position_id}")
                return True
            else:
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
            logger.error(f"Error rebalancing position: {e}")
            self.log_action(
                'rebalance_error',
                current_price,
                status='error',
                notes=str(e)
            )
            return False
    
    def run_cycle(self):
        """Run one cycle of the bot"""
        try:
            # Get current price
            current_price = self.price_oracle.get_current_price()
            if current_price is None:
                logger.error("Failed to get current price")
                return
            
            logger.info(f"Current ETH/USDC price: {current_price:.6f}")
            
            # Add price to strategy
            self.strategy.add_price(current_price)
            
            # Check if rebalancing is needed
            should_rebalance, reason = self.strategy.should_rebalance(
                current_price, self.current_position_range
            )
            
            logger.info(f"Rebalance check: {should_rebalance} - {reason}")
            
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
                
                # Perform rebalancing
                logger.info("Performing rebalance...")
                success = self.rebalance_position(current_price)
                
                if success:
                    logger.info("Rebalance completed successfully")
                else:
                    logger.error("Rebalance failed")
            else:
                # Log price update
                self.log_action(
                    'price_update',
                    current_price,
                    status='no_action',
                    notes=reason
                )
            
            # Log strategy statistics
            stats = self.strategy.get_statistics()
            logger.debug(f"Strategy stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in run cycle: {e}")
    
    def run(self):
        """Main bot loop"""
        logger.info("Starting Uniswap V3 Liquidity Management Bot")
        logger.info(f"Check interval: {settings.CHECK_INTERVAL_MINUTES} minutes")
        logger.info(f"Bollinger Bands: {settings.BOLLINGER_WINDOW} periods, {settings.BOLLINGER_STD_DEV} std dev")
        
        # Log pool info
        pool_info = self.price_oracle.get_pool_info()
        logger.info(f"Pool info: {pool_info}")
        
        self.is_running = True
        
        try:
            while self.is_running:
                logger.info(f"--- Starting cycle at {datetime.now()} ---")
                
                self.run_cycle()
                
                logger.info(f"Cycle completed. Sleeping for {settings.CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping bot...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.is_running = False

def main():
    """Main entry point"""
    bot = UniswapV3Bot()
    bot.run()

if __name__ == "__main__":
    main() 