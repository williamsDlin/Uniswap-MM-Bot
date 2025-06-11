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
        
        # Initialize strategy for range calculations (not decision making)
        self.strategy = BollingerBandsStrategy(
            window=settings.BOLLINGER_WINDOW,
            std_dev=settings.BOLLINGER_STD_DEV
        )
        
        # Initialize RL agent for rebalancing decisions
        self.rl_agent = RebalanceRLAgent(
            state_dim=8,
            action_dim=5,
            learning_rate=3e-4,
            agent_name="rebalance_ppo_agent"
        )
        self.rl_agent.load_model()  # Try to load existing model
        
        # Enhanced CSV logging
        self.csv_filename = f"bot_b_rl_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._initialize_csv()
        
        # Bot state tracking
        self.current_position_range: Optional[Tuple[float, float]] = None
        self.last_rebalance_time = datetime.now()
        self.last_fees_earned = 0.0
        self.last_state = None
        self.last_action = None
        self.last_logprob = None
        self.is_running = False
        self.cycle_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'total_rebalances': 0,
            'rl_decisions': 0,
            'total_fees_earned': 0.0,
            'total_gas_spent': 0.0,
            'successful_positions': 0,
            'failed_positions': 0,
            'no_action_decisions': 0
        }
        
        # Training mode (for simulation/backtesting)
        self.training_mode = False
        
    def _initialize_csv(self):
        """Initialize enhanced CSV file for logging RL bot actions"""
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'cycle', 'current_price', 'rl_action', 'action_name', 'state_features',
                'position_range_before', 'position_range_after', 'fees_earned', 'gas_cost',
                'reward', 'confidence', 'status', 'notes'
            ])
    
    def log_action(self, action_idx: int, action_name: str, current_price: float, **kwargs):
        """Enhanced action logging with RL information"""
        try:
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.cycle_count,
                    current_price,
                    action_idx,
                    action_name,
                    kwargs.get('state_features', ''),
                    kwargs.get('position_range_before', ''),
                    kwargs.get('position_range_after', ''),
                    kwargs.get('fees_earned', ''),
                    kwargs.get('gas_cost', ''),
                    kwargs.get('reward', ''),
                    kwargs.get('confidence', ''),
                    kwargs.get('status', ''),
                    kwargs.get('notes', '')
                ])
        except Exception as e:
            logger.error(f"Failed to log action to CSV: {e}")
    
    def calculate_reward(self, 
                        old_range: Optional[Tuple[float, float]],
                        new_range: Optional[Tuple[float, float]],
                        fees_earned: float,
                        gas_cost: float,
                        current_price: float,
                        action_taken: int) -> float:
        """
        Calculate reward for RL agent based on action outcomes
        
        Args:
            old_range: Previous position range
            new_range: New position range after action
            fees_earned: Fees earned during this period
            gas_cost: Gas cost of the action
            current_price: Current market price
            action_taken: Action index that was taken
            
        Returns:
            Reward value
        """
        try:
            # Base reward: net profit (fees - gas)
            net_profit = fees_earned - gas_cost
            
            # Action-specific rewards
            if action_taken == 0:  # NO_ACTION
                # Reward for not acting when it wasn't necessary
                if old_range and old_range[0] <= current_price <= old_range[1]:
                    # Price is in range, good decision to not rebalance
                    return net_profit + 0.001  # Small bonus for correct inaction
                else:
                    # Price is out of range, should have acted
                    return net_profit - 0.002  # Penalty for missing rebalance opportunity
            
            else:  # Any rebalancing action
                if new_range is None:
                    # Failed to create new position
                    return -0.01  # Large penalty for failed rebalance
                
                # Reward based on how well the new range captures the current price
                new_lower, new_upper = new_range
                range_width = new_upper - new_lower
                range_center = (new_lower + new_upper) / 2
                
                # Position quality: how well centered the price is
                if new_lower <= current_price <= new_upper:
                    # Price is in range
                    center_distance = abs(current_price - range_center) / (range_width / 2)
                    position_quality = 1.0 - center_distance  # 1.0 = perfectly centered, 0.0 = at edge
                else:
                    # Price is outside range (bad positioning)
                    position_quality = -0.5
                
                # Range width penalty/bonus
                optimal_width = current_price * 0.1  # 10% range is considered optimal
                width_ratio = range_width / optimal_width
                
                if 0.5 <= width_ratio <= 2.0:
                    # Reasonable range width
                    width_bonus = 0.001
                else:
                    # Too narrow or too wide
                    width_penalty = -0.001 * abs(width_ratio - 1.0)
                    width_bonus = width_penalty
                
                # Gas efficiency: reward lower gas usage
                gas_efficiency = max(0, 0.005 - gas_cost)  # Bonus for low gas cost
                
                total_reward = net_profit + position_quality * 0.002 + width_bonus + gas_efficiency
                
                return total_reward
                
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return -0.001  # Small penalty for errors
    
    def execute_rl_action(self, action_idx: int, current_price: float) -> Tuple[bool, float, str]:
        """
        Execute the action recommended by RL agent
        
        Args:
            action_idx: Action index from RL agent
            current_price: Current market price
            
        Returns:
            Tuple of (success, gas_cost, notes)
        """
        action_name = self.rl_agent.get_action_name(action_idx)
        
        try:
            if action_idx == 0:  # NO_ACTION
                return True, 0.0, "No action taken as recommended by RL agent"
            
            # For rebalancing actions, we need to determine the new range
            current_range = self.current_position_range
            
            if current_range is None:
                # No existing position, create new one using default strategy
                lower_price, upper_price = self.strategy.get_optimal_range(current_price)
            else:
                # Modify existing range based on RL action
                current_lower, current_upper = current_range
                range_width = current_upper - current_lower
                range_center = (current_lower + current_upper) / 2
                
                if action_idx == 1:  # REBALANCE_NARROW
                    # Make range 20% narrower
                    new_width = range_width * 0.8
                    lower_price = current_price - new_width / 2
                    upper_price = current_price + new_width / 2
                    
                elif action_idx == 2:  # REBALANCE_WIDE
                    # Make range 50% wider
                    new_width = range_width * 1.5
                    lower_price = current_price - new_width / 2
                    upper_price = current_price + new_width / 2
                    
                elif action_idx == 3:  # REBALANCE_SHIFT_UP
                    # Shift range up by 25% of range width
                    shift = range_width * 0.25
                    lower_price = current_lower + shift
                    upper_price = current_upper + shift
                    
                elif action_idx == 4:  # REBALANCE_SHIFT_DOWN
                    # Shift range down by 25% of range width
                    shift = range_width * 0.25
                    lower_price = current_lower - shift
                    upper_price = current_upper - shift
                    
                else:
                    logger.error(f"Unknown action index: {action_idx}")
                    return False, 0.0, "Unknown action"
            
            # Execute the rebalancing
            success = self.rebalance_position(current_price, lower_price, upper_price)
            gas_cost = 0.01 if success else 0.005  # Estimated gas costs
            
            notes = f"Executed {action_name}: range [{lower_price:.6f}, {upper_price:.6f}]"
            
            return success, gas_cost, notes
            
        except Exception as e:
            logger.error(f"Error executing RL action {action_name}: {e}")
            return False, 0.0, f"Error: {str(e)}"
    
    def rebalance_position(self, current_price: float, lower_price: float, upper_price: float) -> bool:
        """Execute position rebalancing"""
        try:
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
                    self.performance_metrics['failed_positions'] += 1
                    return False
            
            # Convert price range to ticks
            lower_tick, upper_tick = self.price_oracle.get_tick_range_for_price_range(
                lower_price, upper_price
            )
            
            # Approve tokens
            if not self.position_manager.approve_tokens(eth_amount, usdc_amount):
                return False
            
            # Mint new position
            position_id = self.position_manager.mint_position(
                lower_tick,
                upper_tick,
                eth_amount,
                usdc_amount
            )
            
            if position_id:
                self.current_position_range = (lower_price, upper_price)
                self.performance_metrics['total_rebalances'] += 1
                self.performance_metrics['successful_positions'] += 1
                
                logger.info(f"Successfully rebalanced with RL guidance. New position ID: {position_id}")
                return True
            else:
                self.performance_metrics['failed_positions'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error in RL-guided rebalancing: {e}")
            return False
    
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
            
            # Add price to strategy for context
            self.strategy.add_price(current_price)
            
            # Calculate time since last rebalance
            time_since_rebalance = (datetime.now() - self.last_rebalance_time).total_seconds() / 3600
            
            # Get current gas price
            gas_price = self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price, 'gwei')
            
            # Extract state for RL agent
            state = self.rl_agent.extract_state(
                current_price=current_price,
                position_range=self.current_position_range,
                fees_earned=0.001,  # Estimated fees since last cycle
                gas_price_gwei=float(gas_price_gwei),
                time_since_rebalance=time_since_rebalance,
                price_history=self.strategy.price_history[-20:],  # Last 20 prices
                position_value=1.0  # Placeholder
            )
            
            # Get action from RL agent
            if self.training_mode:
                action, logprob = self.rl_agent.select_action_with_logprob(state)
            else:
                action = self.rl_agent.select_action(state, deterministic=False)
                logprob = 0.0
            
            action_name = self.rl_agent.get_action_name(action)
            
            logger.info(f"RL Agent decision: {action_name} (action {action})")
            
            # Execute the action
            old_range = self.current_position_range
            success, gas_cost, notes = self.execute_rl_action(action, current_price)
            new_range = self.current_position_range
            
            # Calculate reward
            fees_earned = 0.001  # Estimated fees
            reward = self.calculate_reward(
                old_range, new_range, fees_earned, gas_cost, current_price, action
            )
            
            # Update performance metrics
            self.performance_metrics['rl_decisions'] += 1
            if action == 0:
                self.performance_metrics['no_action_decisions'] += 1
            
            # Store transition for training
            if self.training_mode and self.last_state is not None:
                self.rl_agent.store_transition(
                    self.last_state, self.last_action, self.last_logprob,
                    reward, state, done=False
                )
                
                # Update agent periodically
                if self.cycle_count % 10 == 0:
                    self.rl_agent.update()
            
            # Log action
            self.log_action(
                action, action_name, current_price,
                state_features=state.tolist(),
                position_range_before=old_range,
                position_range_after=new_range,
                fees_earned=fees_earned,
                gas_cost=gas_cost,
                reward=reward,
                status='success' if success else 'failed',
                notes=notes
            )
            
            # Update tracking variables
            self.last_state = state
            self.last_action = action
            self.last_logprob = logprob
            
            if action != 0:  # If we took a rebalancing action
                self.last_rebalance_time = datetime.now()
            
            # Save model periodically
            if self.cycle_count % 20 == 0:
                self.rl_agent.save_model()
                logger.info("Saved RL agent model")
            
        except Exception as e:
            logger.error(f"Error in RL run cycle: {e}")
    
    def run(self):
        """Main bot loop with RL decision making"""
        logger.info("Starting Bot B: RL-driven Rebalancer")
        logger.info(f"Check interval: {settings.CHECK_INTERVAL_MINUTES} minutes")
        logger.info(f"PPO agent loaded with {self.rl_agent.action_dim} actions")
        
        self.is_running = True
        
        try:
            while self.is_running:
                logger.info(f"--- Starting RL cycle {self.cycle_count + 1} at {datetime.now()} ---")
                
                self.run_cycle()
                
                # Log performance summary
                logger.info(f"Performance summary: {self.performance_metrics}")
                
                logger.info(f"Cycle completed. Sleeping for {settings.CHECK_INTERVAL_MINUTES} minutes...")
                time.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping RL bot...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.is_running = False
        finally:
            # Save final model state
            self.rl_agent.save_model()
            logger.info("Final RL model state saved")

def main():
    """Main entry point for Bot B"""
    bot = RLRebalancerBot()
    bot.run()

if __name__ == "__main__":
    main() 