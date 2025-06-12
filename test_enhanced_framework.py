#!/usr/bin/env python3
"""
Test Script for Enhanced Multi-Agent Framework
"""

import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loader():
    """Test the data loader functionality"""
    print("Testing DataLoader...")
    
    try:
        from data_loader import DataLoader
        
        loader = DataLoader()
        
        # Test with last 2 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        print("Loading market data for {} to {}".format(start_date, end_date))
        data = loader.load_market_data(start_date, end_date)
        
        summary = loader.get_data_summary(data)
        print("✅ DataLoader test passed")
        print("   - Loaded {} data points".format(summary['date_range']['duration_hours']))
        print("   - Price range: ${:.2f} - ${:.2f}".format(
            summary['price_stats']['min'], summary['price_stats']['max']))
        
        return True, data, summary
        
    except Exception as e:
        print("❌ DataLoader test failed: {}".format(e))
        return False, None, None

def test_agents():
    """Test the agent implementations"""
    print("\nTesting Agents...")
    
    try:
        # Create agents directory if it doesn't exist
        os.makedirs('agents', exist_ok=True)
        
        # Create __init__.py if it doesn't exist
        init_file = 'agents/__init__.py'
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Agents package\n")
        
        # Test base agent (create a simple implementation)
        from agents.agent_base import AgentBase
        
        class TestAgent(AgentBase):
            def decide_action(self, market_data):
                return {
                    'action': 'hold',
                    'reason': 'Test agent always holds',
                    'confidence': 0.5
                }
        
        agent = TestAgent("test_agent")
        
        # Test with sample market data
        market_data = {
            'price': 2500.0,
            'gas_price': 30.0,
            'volume': 1000000,
            'volatility': 0.05,
            'timestamp': datetime.now()
        }
        
        agent.update_market_data(market_data)
        action = agent.decide_action(market_data)
        result = agent.execute_action(action, market_data)
        
        print("✅ Base Agent test passed")
        print("   - Action: {}".format(action['action']))
        print("   - Executed: {}".format(result['executed']))
        
        return True
        
    except Exception as e:
        print("❌ Agent test failed: {}".format(e))
        return False

def test_simple_experiment():
    """Test a simple experiment with synthetic agents"""
    print("\nTesting Simple Experiment...")
    
    try:
        from data_loader import DataLoader
        from agents.agent_base import AgentBase
        import pandas as pd
        import numpy as np
        
        # Simple baseline agent
        class SimpleBaselineAgent(AgentBase):
            def __init__(self, agent_id, range_width=0.1):
                super().__init__(agent_id)
                self.range_width = range_width
                
            def decide_action(self, market_data):
                if not self.in_position:
                    return {
                        'action': 'enter',
                        'reason': 'Entering position',
                        'new_range': (0.95, 1.05),
                        'confidence': 0.8
                    }
                
                # Simple rebalancing
                if len(self.price_history) >= 10:
                    recent_prices = self.price_history[-10:]
                    current_price = market_data['price']
                    avg_price = np.mean(recent_prices)
                    
                    if abs(current_price - avg_price) / avg_price > 0.05:  # 5% deviation
                        return {
                            'action': 'rebalance',
                            'reason': 'Price deviated from average',
                            'new_range': (0.95, 1.05),
                            'confidence': 0.7
                        }
                
                return {
                    'action': 'hold',
                    'reason': 'Price stable',
                    'confidence': 0.6
                }
        
        # Create agents
        agent1 = SimpleBaselineAgent("baseline")
        agent2 = SimpleBaselineAgent("conservative", range_width=0.15)
        
        # Load some market data
        loader = DataLoader()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        market_data = loader.load_market_data(start_date, end_date)
        price_data = market_data['price']
        gas_data = market_data['gas']
        
        # Run simple simulation
        agents = [agent1, agent2]
        results = {}
        
        # Simulate first 24 steps (1 day)
        max_steps = min(24, len(price_data))
        
        for step in range(max_steps):
            step_data = {
                'price': price_data.iloc[step]['price'],
                'gas_price': gas_data.iloc[step]['gas_price'],
                'volume': price_data.iloc[step]['volume'],
                'volatility': 0.05,  # Default
                'timestamp': price_data.index[step]
            }
            
            for agent in agents:
                agent.update_market_data(step_data)
                action = agent.decide_action(step_data)
                agent.execute_action(action, step_data)
        
        # Get results
        for agent in agents:
            metrics = agent.get_performance_metrics()
            results[agent.agent_id] = metrics
        
        print("✅ Simple Experiment test passed")
        print("   - Simulated {} steps".format(max_steps))
        
        for agent_id, metrics in results.items():
            print("   - {}: ROI {:.2f}%, {} transactions".format(
                agent_id, metrics.get('roi', 0), metrics.get('executed_transactions', 0)))
        
        return True
        
    except Exception as e:
        print("❌ Simple Experiment test failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Multi-Agent Framework")
    print("=" * 50)
    
    # Test 1: Data Loader
    data_test_passed, market_data, summary = test_data_loader()
    
    # Test 2: Agents
    agent_test_passed = test_agents()
    
    # Test 3: Simple Experiment
    experiment_test_passed = test_simple_experiment()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("DataLoader: {}".format("✅ PASS" if data_test_passed else "❌ FAIL"))
    print("Agents: {}".format("✅ PASS" if agent_test_passed else "❌ FAIL"))
    print("Experiment: {}".format("✅ PASS" if experiment_test_passed else "❌ FAIL"))
    
    all_passed = data_test_passed and agent_test_passed and experiment_test_passed
    
    if all_passed:
        print("\n🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python3 enhanced_experiment_runner.py' for full experiments")
        print("2. Run 'streamlit run dashboard.py' for interactive dashboard")
        print("3. Check the generated data files and visualizations")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 