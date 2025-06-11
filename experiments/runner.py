"""
Experiment Runner for Multi-Agent Bot Comparison
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ml.bandit_strategy import LinUCBBanditAgent
from ml.rebalance_rl_agent import RebalanceRLAgent
from ml.gas_predictor import GasPredictorAgent
from strategies.bollinger import BollingerBandsStrategy

class BotSimulator:
    """Simulates a single bot's behavior"""
    
    def __init__(self, bot_id: str, bot_type: str):
        self.bot_id = bot_id
        self.bot_type = bot_type
        self.capital = 1.0
        self.transactions = []
        
        # Initialize strategy
        self.strategy = BollingerBandsStrategy()
        
        # Initialize ML components based on type
        if bot_type == "adaptive_bollinger":
            self.bandit = LinUCBBanditAgent(agent_name=f"bandit_{bot_id}")
        elif bot_type == "rl_rebalancer":
            self.rl_agent = RebalanceRLAgent(agent_name=f"rl_{bot_id}")
        elif bot_type == "gas_aware":
            self.gas_predictor = GasPredictorAgent(agent_name=f"gas_{bot_id}")
    
    def step(self, price: float, gas_price: float, step_num: int) -> Dict:
        """Execute one simulation step"""
        # Update strategy
        self.strategy.add_price(price)
        
        # Make decision based on bot type
        action = self._make_decision(price, gas_price)
        
        # Calculate performance
        fees = 0.0001 if action == "rebalance" else 0.0
        gas_cost = 0.01 * (gas_price / 30) if action == "rebalance" else 0.0
        
        self.capital += fees - gas_cost
        roi = (self.capital - 1.0) * 100
        
        # Record transaction
        transaction = {
            'step': step_num,
            'bot_id': self.bot_id,
            'bot_type': self.bot_type,
            'price': price,
            'gas_price': gas_price,
            'action': action,
            'capital': self.capital,
            'roi': roi,
            'fees': fees,
            'gas_cost': gas_cost
        }
        
        self.transactions.append(transaction)
        return transaction
    
    def _make_decision(self, price: float, gas_price: float) -> str:
        """Make bot-specific decision"""
        # Simple decision logic for demo
        should_rebalance, _ = self.strategy.should_rebalance(price, None)
        
        if not should_rebalance:
            return "hold"
        
        if self.bot_type == "adaptive_bollinger":
            # Use bandit for parameter selection
            context = np.array([price/2000, gas_price/30, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            window, std_dev = self.bandit.select_action(context)
            return f"adaptive_rebalance_w{window}_s{std_dev:.1f}"
            
        elif self.bot_type == "rl_rebalancer":
            # Use RL agent
            state = self.rl_agent.extract_state(
                current_price=price,
                position_range=None,
                fees_earned=0.001,
                gas_price_gwei=gas_price,
                time_since_rebalance=1.0,
                price_history=[price] * 10
            )
            action_idx = self.rl_agent.select_action(state)
            return f"rl_action_{action_idx}"
            
        elif self.bot_type == "gas_aware":
            # Use gas predictor
            self.gas_predictor.add_gas_price_observation(gas_price)
            should_execute, reason, confidence = self.gas_predictor.should_execute_now(
                gas_price, 50.0, 6
            )
            return "gas_optimized_rebalance" if should_execute else "gas_delayed"
        
        return "baseline_rebalance"

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self):
        self.bots = {}
        self.results = {}
    
    def add_bot(self, bot_id: str, bot_type: str):
        """Add a bot to the experiment"""
        self.bots[bot_id] = BotSimulator(bot_id, bot_type)
        print(f"Added bot: {bot_id} ({bot_type})")
    
    def run_experiment(self, num_steps: int = 1000):
        """Run the experiment"""
        print(f"Running experiment with {len(self.bots)} bots for {num_steps} steps...")
        
        # Generate market data
        np.random.seed(42)
        prices = []
        gas_prices = []
        current_price = 2000.0
        
        for i in range(num_steps):
            # Generate price (random walk)
            change = np.random.normal(0, 0.01)
            current_price *= (1 + change)
            prices.append(current_price)
            
            # Generate gas price (daily pattern)
            hour = (i * 5 / 60) % 24  # 5-minute intervals
            base_gas = 20 + 30 * np.sin(hour * np.pi / 12)
            gas_price = max(10, base_gas + np.random.normal(0, 5))
            gas_prices.append(gas_price)
        
        # Run simulation
        all_transactions = []
        start_time = time.time()
        
        for step in range(num_steps):
            price = prices[step]
            gas_price = gas_prices[step]
            
            for bot_id, bot in self.bots.items():
                transaction = bot.step(price, gas_price, step)
                all_transactions.append(transaction)
            
            if step % 100 == 0:
                print(f"Completed step {step}/{num_steps}")
        
        duration = time.time() - start_time
        
        # Collect results
        self.results = {
            'experiment_timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'num_steps': num_steps,
            'market_data': {
                'initial_price': prices[0],
                'final_price': prices[-1],
                'price_change': ((prices[-1] - prices[0]) / prices[0]) * 100,
                'avg_gas_price': np.mean(gas_prices)
            },
            'bot_performance': {}
        }
        
        # Calculate bot statistics
        for bot_id, bot in self.bots.items():
            df = pd.DataFrame(bot.transactions)
            
            self.results['bot_performance'][bot_id] = {
                'bot_type': bot.bot_type,
                'final_roi': df['roi'].iloc[-1],
                'final_capital': df['capital'].iloc[-1],
                'total_rebalances': len(df[df['action'].str.contains('rebalance')]),
                'total_fees': df['fees'].sum(),
                'total_gas_cost': df['gas_cost'].sum(),
                'net_profit': df['fees'].sum() - df['gas_cost'].sum()
            }
        
        print(f"Experiment completed in {duration:.2f} seconds")
        return self.results
    
    def save_results(self, filename: str = None):
        """Save results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
        return filename
    
    def save_transaction_data(self, filename: str = None):
        """Save all transaction data to CSV"""
        all_transactions = []
        for bot in self.bots.values():
            all_transactions.extend(bot.transactions)
        
        if all_transactions:
            df = pd.DataFrame(all_transactions)
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"transaction_data_{timestamp}.csv"
            
            df.to_csv(filename, index=False)
            print(f"Transaction data saved to {filename}")
            return filename
        return None

def main():
    """Run demo experiment"""
    runner = ExperimentRunner()
    
    # Add different bot types
    runner.add_bot("adaptive_bollinger", "adaptive_bollinger")
    runner.add_bot("rl_rebalancer", "rl_rebalancer") 
    runner.add_bot("gas_aware", "gas_aware")
    runner.add_bot("baseline", "baseline")
    
    # Run experiment
    results = runner.run_experiment(1000)
    
    # Save results
    runner.save_results()
    runner.save_transaction_data()
    
    # Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    for bot_id, perf in results['bot_performance'].items():
        print(f"\n{bot_id} ({perf['bot_type']}):")
        print(f"  Final ROI: {perf['final_roi']:.2f}%")
        print(f"  Rebalances: {perf['total_rebalances']}")
        print(f"  Net Profit: {perf['net_profit']:.6f}")

if __name__ == "__main__":
    main() 