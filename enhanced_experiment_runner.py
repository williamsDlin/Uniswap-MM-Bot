#!/usr/bin/env python3
"""
Enhanced Multi-Agent Experiment Runner with Real Market Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os

from data_loader import DataLoader
from agents.agent_base import AgentBase
from agents.adaptive_agent import AdaptiveAgent
from agents.gas_aware_agent import GasAwareAgent

class BaselineAgent(AgentBase):
    """Simple baseline agent for comparison"""
    
    def __init__(self, agent_id: str, initial_capital: float = 1.0, range_width: float = 0.12):
        super().__init__(agent_id, initial_capital)
        self.range_width = range_width
        
    def decide_action(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        current_price = market_data['price']
        
        if not self.in_position:
            # Enter position
            lower_bound = 1.0 - (self.range_width / 2)
            upper_bound = 1.0 + (self.range_width / 2)
            
            return {
                'action': 'enter',
                'reason': 'Entering baseline position',
                'new_range': (lower_bound, upper_bound),
                'confidence': 0.7
            }
        
        # Simple rebalancing based on Bollinger Bands
        if len(self.price_history) >= 20:
            recent_prices = self.price_history[-20:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            # Rebalance if price is outside 2 standard deviations
            if current_price < mean_price - 2 * std_price or current_price > mean_price + 2 * std_price:
                lower_bound = 1.0 - (self.range_width / 2)
                upper_bound = 1.0 + (self.range_width / 2)
                
                return {
                    'action': 'rebalance',
                    'reason': 'Price outside 2-sigma range',
                    'new_range': (lower_bound, upper_bound),
                    'confidence': 0.8
                }
        
        return {
            'action': 'hold',
            'reason': 'Price within acceptable range',
            'confidence': 0.6
        }

class EnhancedExperimentRunner:
    """
    Enhanced experiment runner using real market data and sophisticated agents
    """
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()
        self.agents = {}
        self.market_data = None
        self.experiment_results = {}
        self.current_step = 0
        
        print("Enhanced Experiment Runner initialized")
    
    def add_agent(self, agent: AgentBase):
        """Add an agent to the experiment"""
        self.agents[agent.agent_id] = agent
        print("Added agent: {} ({})".format(agent.agent_id, type(agent).__name__))
    
    def load_market_data(self, start_date: str, end_date: str, etherscan_api_key: Optional[str] = None):
        """Load real market data for the experiment"""
        print("Loading market data for experiment...")
        self.market_data = self.data_loader.load_market_data(start_date, end_date, etherscan_api_key)
        
        # Print data summary
        summary = self.data_loader.get_data_summary(self.market_data)
        print("\nMarket Data Summary:")
        print("- Duration: {} hours".format(summary['date_range']['duration_hours']))
        print("- Price range: ${:.2f} - ${:.2f}".format(
            summary['price_stats']['min'], summary['price_stats']['max']))
        print("- Volatility: {:.2f}%".format(summary['price_stats']['volatility'] * 100))
        print("- Gas range: {:.1f} - {:.1f} gwei".format(
            summary['gas_stats']['min'], summary['gas_stats']['max']))
        
        return summary
    
    def run_experiment(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the multi-agent experiment with real market data
        """
        if not self.market_data:
            raise ValueError("No market data loaded. Call load_market_data() first.")
        
        if not self.agents:
            raise ValueError("No agents added. Call add_agent() first.")
        
        print("\nStarting enhanced experiment with {} agents...".format(len(self.agents)))
        
        # Reset all agents
        for agent in self.agents.values():
            agent.reset()
        
        # Get aligned data
        price_data = self.market_data['price']
        gas_data = self.market_data['gas']
        pool_data = self.market_data['pool']
        
        # Determine number of steps
        total_steps = len(price_data)
        if max_steps:
            total_steps = min(total_steps, max_steps)
        
        print("Running {} simulation steps...".format(total_steps))
        
        # Run simulation
        step_results = []
        
        for step in range(total_steps):
            timestamp = price_data.index[step]
            
            # Prepare market data for this step
            market_data = {
                'timestamp': timestamp,
                'price': price_data.iloc[step]['price'],
                'volume': price_data.iloc[step]['volume'],
                'gas_price': gas_data.iloc[step]['gas_price'],
                'volatility': pool_data.iloc[step]['price_volatility'],
                'liquidity_depth': pool_data.iloc[step]['liquidity_depth'],
                'fees_collected': pool_data.iloc[step]['fees_collected']
            }
            
            step_result = self._run_single_step(step, market_data)
            step_results.append(step_result)
            
            # Progress reporting
            if step % 24 == 0:  # Every 24 hours
                print("Step {}/{} ({:.1f}%)".format(step, total_steps, 100 * step / total_steps))
        
        # Calculate final results
        results = self._calculate_experiment_results(step_results)
        self.experiment_results = results
        
        print("\nExperiment completed!")
        self._print_results_summary(results)
        
        return results
    
    def _run_single_step(self, step: int, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single simulation step for all agents"""
        step_result = {
            'step': step,
            'timestamp': market_data['timestamp'],
            'market_data': market_data,
            'agent_actions': {},
            'agent_states': {}
        }
        
        for agent_id, agent in self.agents.items():
            # Update agent's market view
            agent.update_market_data(market_data)
            
            # Get agent's decision
            action = agent.decide_action(market_data)
            
            # Execute action
            execution_result = agent.execute_action(action, market_data)
            
            # Record results
            step_result['agent_actions'][agent_id] = {
                'action': action,
                'execution': execution_result
            }
            
            # Get agent state
            if hasattr(agent, 'get_strategy_state'):
                step_result['agent_states'][agent_id] = agent.get_strategy_state()
        
        return step_result
    
    def _calculate_experiment_results(self, step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive experiment results"""
        results = {
            'experiment_metadata': {
                'total_steps': len(step_results),
                'start_time': step_results[0]['timestamp'] if step_results else None,
                'end_time': step_results[-1]['timestamp'] if step_results else None,
                'agents': list(self.agents.keys())
            },
            'agent_performance': {},
            'market_summary': {},
            'step_by_step': step_results
        }
        
        # Calculate performance for each agent
        for agent_id, agent in self.agents.items():
            performance = agent.get_performance_metrics()
            
            # Add agent-specific metrics
            if hasattr(agent, 'get_gas_optimization_metrics'):
                gas_metrics = agent.get_gas_optimization_metrics()
                performance.update(gas_metrics)
            
            if hasattr(agent, 'get_strategy_state'):
                strategy_state = agent.get_strategy_state()
                performance['strategy_state'] = strategy_state
            
            results['agent_performance'][agent_id] = performance
        
        # Market summary
        if step_results:
            first_price = step_results[0]['market_data']['price']
            last_price = step_results[-1]['market_data']['price']
            
            all_prices = [s['market_data']['price'] for s in step_results]
            all_gas = [s['market_data']['gas_price'] for s in step_results]
            
            results['market_summary'] = {
                'price_change_pct': ((last_price - first_price) / first_price) * 100,
                'price_volatility': np.std(np.diff(all_prices) / all_prices[:-1]) * np.sqrt(24) * 100,
                'avg_gas_price': np.mean(all_gas),
                'max_gas_price': np.max(all_gas),
                'gas_volatility': np.std(all_gas)
            }
        
        return results
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of experiment results"""
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        
        # Market summary
        market = results['market_summary']
        print("Market Performance:")
        print("  Price change: {:.2f}%".format(market['price_change_pct']))
        print("  Price volatility: {:.2f}%".format(market['price_volatility']))
        print("  Average gas: {:.1f} gwei".format(market['avg_gas_price']))
        
        print("\nAgent Performance:")
        print("{:15} | {:>8} | {:>6} | {:>8} | {:>6}".format(
            "Agent", "ROI (%)", "Txns", "Gas Cost", "Exec %"))
        print("-" * 60)
        
        for agent_id, perf in results['agent_performance'].items():
            roi = perf.get('roi', 0)
            txns = perf.get('executed_transactions', 0)
            gas_cost = perf.get('total_gas_cost', 0)
            exec_rate = perf.get('execution_rate', 0) * 100
            
            print("{:15} | {:>8.2f} | {:>6d} | {:>8.4f} | {:>6.1f}".format(
                agent_id, roi, txns, gas_cost, exec_rate))
        
        # Special metrics for gas-aware agents
        gas_aware_agents = [aid for aid, perf in results['agent_performance'].items() 
                           if 'total_delays' in perf]
        
        if gas_aware_agents:
            print("\nGas Optimization Metrics:")
            print("{:15} | {:>8} | {:>10} | {:>10}".format(
                "Agent", "Delays", "Avg Delay", "Gas Saved"))
            print("-" * 50)
            
            for agent_id in gas_aware_agents:
                perf = results['agent_performance'][agent_id]
                delays = perf.get('total_delays', 0)
                avg_delay = perf.get('avg_delay_duration', 0)
                gas_saved = perf.get('total_gas_saved', 0)
                
                print("{:15} | {:>8d} | {:>10.1f} | {:>10.2f}".format(
                    agent_id, delays, avg_delay, gas_saved))
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save experiment results to JSON file"""
        if not self.experiment_results:
            raise ValueError("No experiment results to save. Run experiment first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "enhanced_experiment_results_{}.json".format(timestamp)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        # Deep convert the results
        json_results = json.loads(json.dumps(self.experiment_results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print("Results saved to: {}".format(filename))
        return filename
    
    def export_transaction_data(self, filename: Optional[str] = None) -> str:
        """Export detailed transaction data to CSV"""
        if not self.agents:
            raise ValueError("No agents available for export.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "enhanced_transactions_{}.csv".format(timestamp)
        
        all_transactions = []
        
        for agent_id, agent in self.agents.items():
            for i, txn in enumerate(agent.transaction_history):
                txn_record = {
                    'step': i,
                    'agent_id': agent_id,
                    'agent_type': type(agent).__name__,
                    **txn
                }
                all_transactions.append(txn_record)
        
        df = pd.DataFrame(all_transactions)
        df.to_csv(filename, index=False)
        
        print("Transaction data exported to: {}".format(filename))
        return filename

def main():
    """Demo of enhanced experiment runner"""
    # Initialize runner
    runner = EnhancedExperimentRunner()
    
    # Add agents
    runner.add_agent(BaselineAgent("baseline", range_width=0.12))
    runner.add_agent(AdaptiveAgent("adaptive", base_range_width=0.1, volatility_multiplier=2.0))
    runner.add_agent(GasAwareAgent("gas_aware", gas_percentile_threshold=80))
    runner.add_agent(AdaptiveAgent("aggressive", base_range_width=0.08, volatility_multiplier=3.0))
    
    # Load market data (last 3 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    summary = runner.load_market_data(start_date, end_date)
    
    # Run experiment
    results = runner.run_experiment(max_steps=72)  # 3 days of hourly data
    
    # Save results
    results_file = runner.save_results()
    transactions_file = runner.export_transaction_data()
    
    return results, results_file, transactions_file

if __name__ == "__main__":
    main() 