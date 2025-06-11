"""
Comprehensive Experiment Runner for Multi-Agent Uniswap V3 Bot Framework
Simulates all bots with market data and generates detailed comparisons
"""

import pandas as pd
import numpy as np
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import concurrent.futures
import asyncio

from ml.bandit_strategy import LinUCBBanditAgent
from ml.rebalance_rl_agent import RebalanceRLAgent
from ml.gas_predictor import GasPredictorAgent
from strategies.adaptive_bollinger import AdaptiveBollingerBandsStrategy
from strategies.bollinger import BollingerBandsStrategy

logger = logging.getLogger(__name__)

class BotSimulator:
    """Simulates a single bot's behavior over time"""
    
    def __init__(self, bot_id: str, bot_type: str, initial_capital: float = 1.0):
        self.bot_id = bot_id
        self.bot_type = bot_type
        self.initial_capital = initial_capital
        
        # Performance tracking
        self.capital = initial_capital
        self.positions = []
        self.transactions = []
        self.performance_history = []
        
        # Initialize bot-specific components
        self._initialize_bot_components()
        
    def _initialize_bot_components(self):
        """Initialize bot-specific ML components"""
        if self.bot_type == "adaptive_bollinger":
            self.strategy = AdaptiveBollingerBandsStrategy(
                initial_window=20,
                initial_std_dev=2.0,
                bandit_alpha=1.0
            )
            
        elif self.bot_type == "rl_rebalancer":
            self.strategy = BollingerBandsStrategy(window=20, std_dev=2.0)
            self.rl_agent = RebalanceRLAgent(agent_name=f"rl_agent_{self.bot_id}")
            
        elif self.bot_type == "gas_aware":
            self.strategy = BollingerBandsStrategy(window=20, std_dev=2.0)
            self.gas_predictor = GasPredictorAgent(agent_name=f"gas_predictor_{self.bot_id}")
            
        elif self.bot_type == "baseline":
            self.strategy = BollingerBandsStrategy(window=20, std_dev=2.0)
            
        else:
            raise ValueError(f"Unknown bot type: {self.bot_type}")
    
    def step(self, market_data: Dict, step_number: int) -> Dict:
        """Execute one simulation step"""
        current_price = market_data['price']
        gas_price = market_data.get('gas_price', 30.0)
        timestamp = market_data.get('timestamp', datetime.now())
        
        # Update strategy with new price
        if hasattr(self.strategy, 'add_price'):
            self.strategy.add_price(current_price)
        
        # Make decision based on bot type
        action_taken = self._make_decision(current_price, gas_price, market_data)
        
        # Calculate performance
        performance = self._calculate_step_performance(current_price, gas_price, action_taken)
        
        # Record transaction
        transaction = {
            'step': step_number,
            'timestamp': timestamp,
            'bot_id': self.bot_id,
            'action': action_taken['action'],
            'current_price': current_price,
            'gas_price': gas_price,
            'position_range': action_taken.get('position_range'),
            'capital': self.capital,
            'fees_earned': performance['fees_earned'],
            'gas_cost': performance['gas_cost'],
            'roi': performance['roi'],
            'confidence': action_taken.get('confidence', 0.0),
            'reason': action_taken.get('reason', '')
        }
        
        self.transactions.append(transaction)
        self.performance_history.append(performance)
        
        return transaction
    
    def _make_decision(self, current_price: float, gas_price: float, market_data: Dict) -> Dict:
        """Make bot-specific decision"""
        current_position_range = self.positions[-1]['range'] if self.positions else None
        
        if self.bot_type == "adaptive_bollinger":
            return self._adaptive_bollinger_decision(current_price, current_position_range, market_data)
            
        elif self.bot_type == "rl_rebalancer":
            return self._rl_decision(current_price, current_position_range, gas_price, market_data)
            
        elif self.bot_type == "gas_aware":
            return self._gas_aware_decision(current_price, current_position_range, gas_price, market_data)
            
        else:  # baseline
            return self._baseline_decision(current_price, current_position_range)
    
    def _adaptive_bollinger_decision(self, price: float, current_range: Optional[tuple], market_data: Dict) -> Dict:
        """Adaptive Bollinger Band decision with bandit learning"""
        # Check if rebalancing is needed
        should_rebalance, reason = self.strategy.should_rebalance(price, current_range)
        
        if should_rebalance:
            # Get optimal range using adaptive parameters
            lower_price, upper_price = self.strategy.get_optimal_range(price)
            
            # Record position
            self.positions.append({
                'timestamp': market_data.get('timestamp', datetime.now()),
                'range': (lower_price, upper_price),
                'price': price,
                'window': getattr(self.strategy, 'current_window', 20),
                'std_dev': getattr(self.strategy, 'current_std_dev', 2.0)
            })
            
            return {
                'action': 'adaptive_rebalance',
                'position_range': (lower_price, upper_price),
                'confidence': 0.8,
                'reason': f"Adaptive rebalance: {reason}, window={getattr(self.strategy, 'current_window', 20)}"
            }
        
        return {
            'action': 'hold',
            'position_range': current_range,
            'confidence': 0.6,
            'reason': f"No rebalance needed: {reason}"
        }
    
    def _rl_decision(self, price: float, current_range: Optional[tuple], gas_price: float, market_data: Dict) -> Dict:
        """RL-based rebalancing decision"""
        # Extract state for RL agent
        state = self.rl_agent.extract_state(
            current_price=price,
            position_range=current_range,
            fees_earned=0.001,
            gas_price_gwei=gas_price,
            time_since_rebalance=1.0,
            price_history=getattr(self.strategy, 'price_history', [price] * 10)[-10:]
        )
        
        # Get action from RL agent
        action_idx = self.rl_agent.select_action(state)
        action_name = self.rl_agent.get_action_name(action_idx)
        
        if action_idx == 0:  # NO_ACTION
            return {
                'action': 'rl_hold',
                'position_range': current_range,
                'confidence': 0.7,
                'reason': f"RL decision: {action_name}",
                'rl_action': action_name
            }
        else:
            # Calculate new range based on RL action
            if current_range is None:
                lower_price, upper_price = self.strategy.get_optimal_range(price)
            else:
                lower_price, upper_price = self._modify_range_for_rl_action(
                    current_range, action_idx, price
                )
            
            # Record position
            self.positions.append({
                'timestamp': market_data.get('timestamp', datetime.now()),
                'range': (lower_price, upper_price),
                'price': price,
                'rl_action': action_name
            })
            
            return {
                'action': f'rl_{action_name.lower()}',
                'position_range': (lower_price, upper_price),
                'confidence': 0.8,
                'reason': f"RL decision: {action_name}",
                'rl_action': action_name
            }
    
    def _gas_aware_decision(self, price: float, current_range: Optional[tuple], gas_price: float, market_data: Dict) -> Dict:
        """Gas-aware execution decision"""
        # Update gas predictor
        self.gas_predictor.add_gas_price_observation(gas_price)
        
        # Check if rebalancing is needed
        should_rebalance, reason = self.strategy.should_rebalance(price, current_range)
        
        if should_rebalance:
            # Check gas conditions
            should_execute, gas_reason, confidence = self.gas_predictor.should_execute_now(
                current_gas_price=gas_price,
                max_gas_price=50.0,
                urgency_hours=6
            )
            
            if should_execute:
                lower_price, upper_price = self.strategy.get_optimal_range(price)
                
                self.positions.append({
                    'timestamp': market_data.get('timestamp', datetime.now()),
                    'range': (lower_price, upper_price),
                    'price': price,
                    'gas_price': gas_price
                })
                
                return {
                    'action': 'gas_optimized_rebalance',
                    'position_range': (lower_price, upper_price),
                    'confidence': confidence,
                    'reason': f"Gas-optimized execution: {gas_reason}"
                }
            else:
                return {
                    'action': 'gas_delayed',
                    'position_range': current_range,
                    'confidence': confidence,
                    'reason': f"Gas-delayed execution: {gas_reason}"
                }
        
        return {
            'action': 'hold',
            'position_range': current_range,
            'confidence': 0.6,
            'reason': f"No rebalance needed: {reason}"
        }
    
    def _baseline_decision(self, price: float, current_range: Optional[tuple]) -> Dict:
        """Simple baseline strategy decision"""
        should_rebalance, reason = self.strategy.should_rebalance(price, current_range)
        
        if should_rebalance:
            lower_price, upper_price = self.strategy.get_optimal_range(price)
            
            self.positions.append({
                'timestamp': datetime.now(),
                'range': (lower_price, upper_price),
                'price': price
            })
            
            return {
                'action': 'baseline_rebalance',
                'position_range': (lower_price, upper_price),
                'confidence': 0.5,
                'reason': f"Baseline rebalance: {reason}"
            }
        
        return {
            'action': 'hold',
            'position_range': current_range,
            'confidence': 0.5,
            'reason': f"No rebalance needed: {reason}"
        }
    
    def _modify_range_for_rl_action(self, current_range: tuple, action_idx: int, current_price: float) -> tuple:
        """Modify range based on RL action"""
        lower, upper = current_range
        range_width = upper - lower
        
        if action_idx == 1:  # REBALANCE_NARROW
            new_width = range_width * 0.8
            return (current_price - new_width/2, current_price + new_width/2)
        elif action_idx == 2:  # REBALANCE_WIDE
            new_width = range_width * 1.5
            return (current_price - new_width/2, current_price + new_width/2)
        elif action_idx == 3:  # REBALANCE_SHIFT_UP
            shift = range_width * 0.25
            return (lower + shift, upper + shift)
        elif action_idx == 4:  # REBALANCE_SHIFT_DOWN
            shift = range_width * 0.25
            return (lower - shift, upper - shift)
        else:
            return current_range
    
    def _calculate_step_performance(self, price: float, gas_price: float, action: Dict) -> Dict:
        """Calculate performance metrics for this step"""
        # Estimate fees earned (simplified)
        fees_earned = 0.0
        if self.positions and action['action'] != 'hold':
            # Estimate fees based on position efficiency
            position = self.positions[-1]
            if position['range'][0] <= price <= position['range'][1]:
                fees_earned = 0.0003  # 0.03% fee estimate for in-range position
        
        # Calculate gas cost
        gas_cost = 0.0
        if 'rebalance' in action['action']:
            gas_cost = 0.01 * (gas_price / 30.0)  # Scale with gas price
        
        # Update capital
        net_change = fees_earned - gas_cost
        self.capital += net_change
        
        # Calculate ROI
        roi = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'fees_earned': fees_earned,
            'gas_cost': gas_cost,
            'net_change': net_change,
            'capital': self.capital,
            'roi': roi
        }
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for this bot"""
        if not self.transactions:
            return {}
        
        df = pd.DataFrame(self.transactions)
        
        total_rebalances = len(df[df['action'].str.contains('rebalance')])
        total_fees = df['fees_earned'].sum()
        total_gas = df['gas_cost'].sum()
        final_roi = df['roi'].iloc[-1] if len(df) > 0 else 0
        
        # Calculate position effectiveness
        in_range_count = 0
        total_position_periods = 0
        
        for _, row in df.iterrows():
            if pd.notna(row.get('position_range')) and row['position_range'] is not None:
                try:
                    lower, upper = row['position_range']
                    if lower <= row['current_price'] <= upper:
                        in_range_count += 1
                    total_position_periods += 1
                except (TypeError, ValueError):
                    continue
        
        position_effectiveness = (in_range_count / total_position_periods * 100) if total_position_periods > 0 else 0
        
        return {
            'bot_id': self.bot_id,
            'bot_type': self.bot_type,
            'total_steps': len(df),
            'total_rebalances': total_rebalances,
            'rebalance_frequency': total_rebalances / len(df) if len(df) > 0 else 0,
            'total_fees_earned': total_fees,
            'total_gas_cost': total_gas,
            'net_profit': total_fees - total_gas,
            'final_roi': final_roi,
            'final_capital': self.capital,
            'position_effectiveness': position_effectiveness,
            'avg_confidence': df['confidence'].mean() if 'confidence' in df else 0
        }

class MarketDataGenerator:
    """Generates realistic market data for simulation"""
    
    def __init__(self, initial_price: float = 2000.0, volatility: float = 0.02):
        self.initial_price = initial_price
        self.volatility = volatility
        self.current_price = initial_price
        
    def generate_price_series(self, num_steps: int, time_interval_minutes: int = 5) -> List[Dict]:
        """Generate a realistic price series with trends and volatility"""
        np.random.seed(42)  # For reproducible results
        
        market_data = []
        timestamp = datetime.now()
        
        for i in range(num_steps):
            # Generate price using geometric Brownian motion with trends
            dt = time_interval_minutes / (60 * 24)  # Convert to days
            
            # Add daily and weekly patterns
            hour_of_day = (timestamp.hour + timestamp.minute / 60) / 24
            day_of_week = timestamp.weekday() / 7
            
            # Volatility varies by time (higher during US/EU hours)
            time_volatility = self.volatility * (1 + 0.5 * np.sin(hour_of_day * 2 * np.pi))
            
            # Trend component (slight upward bias)
            trend = 0.0001 * dt
            
            # Random walk component
            random_change = np.random.normal(trend, time_volatility * np.sqrt(dt))
            
            # Update price
            self.current_price *= (1 + random_change)
            
            # Generate gas price (correlated with network activity)
            base_gas = 20 + 30 * np.sin(hour_of_day * 2 * np.pi)  # Daily pattern
            gas_volatility = np.random.normal(0, 5)
            gas_price = max(10, base_gas + gas_volatility)
            
            market_data.append({
                'step': i,
                'timestamp': timestamp,
                'price': self.current_price,
                'gas_price': gas_price,
                'hour_of_day': hour_of_day,
                'volatility': time_volatility
            })
            
            timestamp += timedelta(minutes=time_interval_minutes)
        
        return market_data

class ExperimentRunner:
    """Main experiment runner for multi-bot comparison"""
    
    def __init__(self, experiment_name: str = "multi_bot_comparison"):
        self.experiment_name = experiment_name
        self.bots = {}
        self.market_data = []
        self.results = {}
        
        # Experiment configuration
        self.num_steps = 2016  # 7 days * 24 hours * 12 (5-minute intervals)
        self.time_interval_minutes = 5
        self.initial_capital = 1.0
        
        logger.info(f"Initialized experiment: {experiment_name}")
    
    def add_bot(self, bot_id: str, bot_type: str) -> None:
        """Add a bot to the experiment"""
        self.bots[bot_id] = BotSimulator(bot_id, bot_type, self.initial_capital)
        logger.info(f"Added bot: {bot_id} ({bot_type})")
    
    def generate_market_data(self, initial_price: float = 2000.0, volatility: float = 0.02) -> None:
        """Generate market data for the experiment"""
        generator = MarketDataGenerator(initial_price, volatility)
        self.market_data = generator.generate_price_series(self.num_steps, self.time_interval_minutes)
        logger.info(f"Generated {len(self.market_data)} market data points")
    
    def run_experiment(self, parallel: bool = True) -> Dict:
        """Run the multi-bot experiment"""
        logger.info(f"Starting experiment with {len(self.bots)} bots...")
        
        if not self.market_data:
            self.generate_market_data()
        
        start_time = time.time()
        
        if parallel:
            self._run_parallel_simulation()
        else:
            self._run_sequential_simulation()
        
        duration = time.time() - start_time
        
        # Collect results
        self.results = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'num_steps': self.num_steps,
            'time_interval_minutes': self.time_interval_minutes,
            'initial_capital': self.initial_capital,
            'bots': {},
            'market_summary': self._get_market_summary()
        }
        
        # Get bot results
        for bot_id, bot in self.bots.items():
            self.results['bots'][bot_id] = {
                'summary_stats': bot.get_summary_stats(),
                'transactions': bot.transactions[-100:],  # Last 100 transactions
                'final_performance': bot.performance_history[-1] if bot.performance_history else {}
            }
        
        logger.info(f"Experiment completed in {duration:.2f} seconds")
        return self.results
    
    def _run_parallel_simulation(self):
        """Run simulation with parallel bot execution"""
        def simulate_bot(bot_item):
            bot_id, bot = bot_item
            bot_transactions = []
            
            for step, market_point in enumerate(self.market_data):
                transaction = bot.step(market_point, step)
                bot_transactions.append(transaction)
                
                # Update ML agents periodically
                if step % 20 == 0 and hasattr(bot, 'rl_agent'):
                    bot.rl_agent.update()
            
            return bot_id, bot_transactions
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.bots)) as executor:
            futures = [executor.submit(simulate_bot, item) for item in self.bots.items()]
            
            for future in concurrent.futures.as_completed(futures):
                bot_id, transactions = future.result()
                logger.info(f"Completed simulation for {bot_id}: {len(transactions)} transactions")
    
    def _run_sequential_simulation(self):
        """Run simulation sequentially (for debugging)"""
        for step, market_point in enumerate(self.market_data):
            for bot_id, bot in self.bots.items():
                bot.step(market_point, step)
            
            if step % 100 == 0:
                logger.info(f"Completed step {step}/{self.num_steps}")
    
    def _get_market_summary(self) -> Dict:
        """Get summary statistics of market data"""
        if not self.market_data:
            return {}
        
        prices = [point['price'] for point in self.market_data]
        gas_prices = [point['gas_price'] for point in self.market_data]
        
        return {
            'initial_price': prices[0],
            'final_price': prices[-1],
            'min_price': min(prices),
            'max_price': max(prices),
            'price_change_percent': ((prices[-1] - prices[0]) / prices[0]) * 100,
            'price_volatility': np.std(np.diff(prices) / prices[:-1]),
            'avg_gas_price': np.mean(gas_prices),
            'max_gas_price': max(gas_prices),
            'min_gas_price': min(gas_prices)
        }
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save experiment results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{self.experiment_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def save_transaction_data(self) -> str:
        """Save all transaction data to CSV for detailed analysis"""
        all_transactions = []
        
        for bot_id, bot in self.bots.items():
            all_transactions.extend(bot.transactions)
        
        if all_transactions:
            df = pd.DataFrame(all_transactions)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transaction_data_{self.experiment_name}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Transaction data saved to {filename}")
            return filename
        
        return ""

def main():
    """Example experiment run"""
    # Create experiment
    experiment = ExperimentRunner("demo_experiment")
    
    # Add different bot types
    experiment.add_bot("adaptive_bollinger", "adaptive_bollinger")
    experiment.add_bot("rl_rebalancer", "rl_rebalancer")
    experiment.add_bot("gas_aware", "gas_aware")
    experiment.add_bot("baseline", "baseline")
    
    # Run experiment
    results = experiment.run_experiment(parallel=True)
    
    # Save results
    experiment.save_results()
    experiment.save_transaction_data()
    
    # Print summary
    print("\n=== EXPERIMENT RESULTS ===")
    for bot_id, bot_data in results['bots'].items():
        stats = bot_data['summary_stats']
        print(f"\n{bot_id} ({stats['bot_type']}):")
        print(f"  ROI: {stats['final_roi']:.2f}%")
        print(f"  Rebalances: {stats['total_rebalances']}")
        print(f"  Net Profit: {stats['net_profit']:.6f}")
        print(f"  Position Effectiveness: {stats['position_effectiveness']:.1f}%")

if __name__ == "__main__":
    main() 