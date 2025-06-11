"""
Bot D: Gas-Aware Executor
Uses SARIMA-based gas price prediction to optimize transaction timing
"""

import time
import csv
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from web3 import Web3

from config.settings import settings
from config.contracts import ETH_USDC_POOL
from price_oracle import PriceOracle
from position_manager import PositionManager
from strategies.bollinger import BollingerBandsStrategy
from ml.gas_predictor import GasPredictorAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_d_gas_aware.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class GasAwareBot:
    """Bot D: Gas-aware executor with transaction timing optimization"""
    
    def __init__(self, bot_id: str = "bot_d"):
        """Initialize the gas-aware bot"""
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
        
        # Initialize strategy for decision making
        self.strategy = BollingerBandsStrategy(
            window=settings.BOLLINGER_WINDOW,
            std_dev=settings.BOLLINGER_STD_DEV
        )
        
        # Initialize gas predictor
        self.gas_predictor = GasPredictorAgent(
            max_history=1000,
            prediction_horizon=12,  # 12 hours ahead
            update_frequency=5,     # Update every 5 observations
            agent_name="gas_predictor_bot_d"
        )
        self.gas_predictor.load_model()
        
        # Enhanced CSV logging
        self.csv_filename = f"bot_d_gas_aware_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._initialize_csv()
        
        # Bot state tracking
        self.current_position_range: Optional[Tuple[float, float]] = None
        self.pending_transactions = []  # Queue of transactions waiting for better gas prices
        self.is_running = False
        self.cycle_count = 0
        
        # Gas-aware settings
        self.max_gas_price_gwei = getattr(settings, 'MAX_GAS_PRICE_GWEI', 50)
        self.gas_savings_threshold = 0.15  # 15% savings threshold
        self.max_wait_hours = 6  # Maximum hours to wait for better gas prices
        
        # Performance tracking
        self.performance_metrics = {
            'total_rebalances': 0,
            'gas_savings_achieved': 0.0,
            'transactions_delayed': 0,
            'transactions_executed_immediately': 0,
            'average_gas_price_paid': 0.0,
            'total_gas_spent': 0.0
        }
        
    def _initialize_csv(self):
        """Initialize enhanced CSV file for logging gas-aware actions"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'cycle', 'action', 'current_price', 'current_gas_gwei', 
                'predicted_min_gas', 'optimal_execution_hour', 'should_execute_now',
                'execution_reason', 'confidence', 'gas_savings_percent', 'position_range',
                'transaction_type', 'gas_used', 'status', 'notes'
            ])
    
    def log_action(self, action: str, current_price: float, **kwargs):
        """Enhanced action logging with gas prediction information"""
        try:
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.cycle_count,
                    action,
                    current_price,
                    kwargs.get('current_gas_gwei', ''),
                    kwargs.get('predicted_min_gas', ''),
                    kwargs.get('optimal_execution_hour', ''),
                    kwargs.get('should_execute_now', ''),
                    kwargs.get('execution_reason', ''),
                    kwargs.get('confidence', ''),
                    kwargs.get('gas_savings_percent', ''),
                    kwargs.get('position_range', ''),
                    kwargs.get('transaction_type', ''),
                    kwargs.get('gas_used', ''),
                    kwargs.get('status', ''),
                    kwargs.get('notes', '')
                ])
        except Exception as e:
            logger.error(f"Failed to log action to CSV: {e}")
    
    def check_and_update_gas_predictions(self) -> None:
        """Update gas price predictions with current data"""
        try:
            # Get current gas price
            current_gas_price = self.web3.eth.gas_price
            current_gas_gwei = self.web3.from_wei(current_gas_price, 'gwei')
            
            # Add observation to gas predictor
            self.gas_predictor.add_gas_price_observation(float(current_gas_gwei))
            
            logger.debug(f"Updated gas predictions with current price: {current_gas_gwei:.2f} Gwei")
            
        except Exception as e:
            logger.error(f"Error updating gas predictions: {e}")
    
    def should_execute_transaction(self, 
                                 transaction_type: str,
                                 urgency_hours: int = None) -> Tuple[bool, str, float, dict]:
        """
        Determine if a transaction should be executed now based on gas predictions
        
        Args:
            transaction_type: Type of transaction ('rebalance', 'approve', 'mint', 'remove')
            urgency_hours: How urgent the transaction is (defaults to max_wait_hours)
            
        Returns:
            Tuple of (should_execute, reason, confidence, gas_info)
        """
        if urgency_hours is None:
            urgency_hours = self.max_wait_hours
        
        try:
            # Get current gas price
            current_gas_price = self.web3.eth.gas_price
            current_gas_gwei = self.web3.from_wei(current_gas_price, 'gwei')
            
            # Get execution recommendation from gas predictor
            should_execute, reason, confidence = self.gas_predictor.should_execute_now(
                current_gas_price=float(current_gas_gwei),
                max_gas_price=self.max_gas_price_gwei,
                urgency_hours=urgency_hours
            )
            
            # Get optimal execution time
            optimal_hour, optimal_price, opt_confidence = self.gas_predictor.get_optimal_execution_time(
                max_hours=urgency_hours
            )
            
            # Calculate potential savings
            gas_savings_percent = 0.0
            if optimal_price < current_gas_gwei:
                gas_savings_percent = ((current_gas_gwei - optimal_price) / current_gas_gwei) * 100
            
            gas_info = {
                'current_gas_gwei': float(current_gas_gwei),
                'optimal_price': optimal_price,
                'optimal_hour': optimal_hour,
                'gas_savings_percent': gas_savings_percent,
                'prediction_confidence': opt_confidence
            }
            
            # Override decision for urgent transactions
            if transaction_type in ['remove', 'emergency'] and current_gas_gwei <= self.max_gas_price_gwei * 1.5:
                should_execute = True
                reason = f"Urgent {transaction_type} transaction, executing despite gas price"
                confidence = 0.8
            
            logger.info(f"Gas execution decision for {transaction_type}: {should_execute} - {reason} "
                       f"(current: {current_gas_gwei:.1f}, optimal: {optimal_price:.1f} in {optimal_hour}h)")
            
            return should_execute, reason, confidence, gas_info
            
        except Exception as e:
            logger.error(f"Error in gas execution decision: {e}")
            # Default to conservative execution
            return True, "Error in gas prediction, executing conservatively", 0.3, {}
    
    def execute_gas_aware_rebalance(self, current_price: float) -> bool:
        """Execute rebalancing with gas-aware timing"""
        try:
            # Check if we should execute now
            should_execute, reason, confidence, gas_info = self.should_execute_transaction(
                'rebalance', urgency_hours=self.max_wait_hours
            )
            
            # Log the decision
            self.log_action(
                'gas_decision_rebalance',
                current_price,
                current_gas_gwei=gas_info.get('current_gas_gwei'),
                predicted_min_gas=gas_info.get('optimal_price'),
                optimal_execution_hour=gas_info.get('optimal_hour'),
                should_execute_now=should_execute,
                execution_reason=reason,
                confidence=confidence,
                gas_savings_percent=gas_info.get('gas_savings_percent'),
                transaction_type='rebalance'
            )
            
            if not should_execute:
                # Add to pending transactions or wait
                logger.info(f"Delaying rebalance due to gas prices: {reason}")
                self.performance_metrics['transactions_delayed'] += 1
                return False
            
            # Execute rebalancing immediately
            logger.info(f"Executing rebalance: {reason}")
            success = self._perform_rebalance(current_price, gas_info)
            
            if success:
                self.performance_metrics['transactions_executed_immediately'] += 1
                
                # Calculate actual gas savings if we waited
                if gas_info.get('gas_savings_percent', 0) > 0:
                    self.performance_metrics['gas_savings_achieved'] += gas_info['gas_savings_percent']
            
            return success
            
        except Exception as e:
            logger.error(f"Error in gas-aware rebalancing: {e}")
            return False
    
    def _perform_rebalance(self, current_price: float, gas_info: dict) -> bool:
        """Perform the actual rebalancing transaction"""
        try:
            # Get optimal range from strategy
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
                    return False
                
                gas_used += 150000  # Estimated gas for removal
            
            # Convert price range to ticks
            lower_tick, upper_tick = self.price_oracle.get_tick_range_for_price_range(
                lower_price, upper_price
            )
            
            # Approve tokens (check gas first)
            should_approve, approve_reason, _, approve_gas_info = self.should_execute_transaction(
                'approve', urgency_hours=1  # Approvals are more urgent
            )
            
            if should_approve:
                if not self.position_manager.approve_tokens(eth_amount, usdc_amount):
                    return False
                gas_used += 100000  # Estimated gas for approvals
            else:
                logger.info(f"Delaying token approval due to gas: {approve_reason}")
                return False
            
            # Mint new position
            position_id = self.position_manager.mint_position(
                lower_tick,
                upper_tick,
                eth_amount,
                usdc_amount
            )
            
            gas_used += 400000  # Estimated gas for minting
            
            if position_id:
                self.current_position_range = (lower_price, upper_price)
                self.performance_metrics['total_rebalances'] += 1
                
                # Calculate gas cost in ETH
                gas_price_wei = self.web3.to_wei(gas_info.get('current_gas_gwei', 30), 'gwei')
                gas_cost_eth = (gas_used * gas_price_wei) / 10**18
                self.performance_metrics['total_gas_spent'] += gas_cost_eth
                
                # Update average gas price
                total_txs = self.performance_metrics['transactions_executed_immediately']
                current_avg = self.performance_metrics['average_gas_price_paid']
                new_avg = ((current_avg * (total_txs - 1)) + gas_info.get('current_gas_gwei', 30)) / total_txs
                self.performance_metrics['average_gas_price_paid'] = new_avg
                
                self.log_action(
                    'rebalance_executed',
                    current_price,
                    position_range=(lower_price, upper_price),
                    gas_used=gas_used,
                    current_gas_gwei=gas_info.get('current_gas_gwei'),
                    status='success',
                    notes=f"Position ID: {position_id}"
                )
                
                logger.info(f"Gas-aware rebalance completed successfully. Position ID: {position_id}, "
                           f"Gas used: {gas_used}, Gas price: {gas_info.get('current_gas_gwei', 'N/A'):.1f} Gwei")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error performing rebalance: {e}")
            return False
    
    def run_cycle(self):
        """Run one cycle of the gas-aware bot"""
        try:
            self.cycle_count += 1
            
            # Update gas predictions with current data
            self.check_and_update_gas_predictions()
            
            # Get current price
            current_price = self.price_oracle.get_current_price()
            if current_price is None:
                logger.error("Failed to get current price")
                return
            
            logger.info(f"Cycle {self.cycle_count}: Current ETH/USDC price: {current_price:.6f}")
            
            # Add price to strategy
            self.strategy.add_price(current_price)
            
            # Check if rebalancing is needed
            should_rebalance, reason = self.strategy.should_rebalance(
                current_price, self.current_position_range
            )
            
            # Get gas statistics
            gas_stats = self.gas_predictor.get_gas_statistics()
            logger.info(f"Gas stats: Current avg: {gas_stats.get('recent_avg_gas', 'N/A'):.1f} Gwei, "
                       f"Model fitted: {gas_stats.get('model_fitted', False)}")
            
            if should_rebalance:
                logger.info(f"Rebalancing needed: {reason}")
                
                # Check safety conditions first
                eth_balance, _ = self.position_manager.get_token_balances()
                if eth_balance < settings.MIN_ETH_BALANCE:
                    logger.warning(f"Insufficient ETH balance: {eth_balance:.6f}")
                    return
                
                # Execute gas-aware rebalancing
                success = self.execute_gas_aware_rebalance(current_price)
                
                if success:
                    logger.info("Gas-aware rebalance completed successfully")
                else:
                    logger.warning("Gas-aware rebalance failed or delayed")
            else:
                # Log price update with gas information
                current_gas_gwei = self.web3.from_wei(self.web3.eth.gas_price, 'gwei')
                
                self.log_action(
                    'price_update',
                    current_price,
                    current_gas_gwei=float(current_gas_gwei),
                    status='no_rebalance_needed',
                    notes=reason
                )
            
            # Save gas predictor model periodically
            if self.cycle_count % 20 == 0:
                self.gas_predictor.save_model()
                logger.info("Saved gas predictor model")
            
        except Exception as e:
            logger.error(f"Error in gas-aware run cycle: {e}")
    
    def run(self):
        """Main bot loop with gas-aware execution"""
        logger.info("Starting Bot D: Gas-Aware Executor")
        logger.info(f"Check interval: {settings.CHECK_INTERVAL_MINUTES} minutes")
        logger.info(f"Max gas price: {self.max_gas_price_gwei} Gwei")
        logger.info(f"Gas savings threshold: {self.gas_savings_threshold*100:.1f}%")
        logger.info(f"Max wait time: {self.max_wait_hours} hours")
        
        # Initialize gas predictor with some initial data
        logger.info("Collecting initial gas price data...")
        for _ in range(5):
            self.check_and_update_gas_predictions()
            time.sleep(2)
        
        self.is_running = True
        
        try:
            while self.is_running:
                logger.info(f"--- Starting gas-aware cycle {self.cycle_count + 1} at {datetime.now()} ---")
                
                self.run_cycle()
                
                # Log performance summary
                logger.info(f"Performance summary: {self.performance_metrics}")
                
                # Log gas prediction stats
                gas_stats = self.gas_predictor.get_gas_statistics()
                if gas_stats.get('model_fitted'):
                    predictions = gas_stats.get('next_6h_predictions', {})
                    if predictions:
                        min_6h = min(predictions.values())
                        max_6h = max(predictions.values())
                        logger.info(f"Gas forecast next 6h: {min_6h:.1f} - {max_6h:.1f} Gwei")
                
                logger.info(f"Cycle completed. Sleeping for {settings.CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping gas-aware bot...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.is_running = False
        finally:
            # Save final gas predictor state
            self.gas_predictor.save_model()
            logger.info("Final gas predictor model saved")

def main():
    """Main entry point for Bot D"""
    bot = GasAwareBot()
    bot.run()

if __name__ == "__main__":
    main() 