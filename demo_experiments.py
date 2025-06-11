#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Bot Demo
"""

from experiments.runner import ExperimentRunner

def main():
    print("Multi-Agent Bot Demo Starting...")

    # Create experiment runner
    runner = ExperimentRunner()

    # Add bots
    runner.add_bot('adaptive', 'adaptive_bollinger')
    runner.add_bot('rl_agent', 'rl_rebalancer')
    runner.add_bot('gas_optimized', 'gas_aware')
    runner.add_bot('baseline', 'baseline')

    print("\nRunning demo experiment (100 steps)...")

    # Run experiment
    results = runner.run_experiment(100)

    print("\n" + "="*60)
    print("DEMO RESULTS:")
    print("="*60)
    
    for bot_id, perf in results['bot_performance'].items():
        roi = perf['final_roi']
        rebalances = perf['total_rebalances']
        net_profit = perf['net_profit']
        print("{:15} | ROI: {:6.2f}% | Rebalances: {:3d} | Net: {:8.6f}".format(bot_id, roi, rebalances, net_profit))

    # Save files
    results_file = runner.save_results()
    transaction_file = runner.save_transaction_data()
    
    print("\nResults saved: {}".format(results_file))
    print("Transactions saved: {}".format(transaction_file))
    
    print("\nDemo completed successfully!")
    
    # Market summary
    market = results['market_data']
    print("\nMarket Summary:")
    print("Price change: {:.2f}%".format(market['price_change']))
    print("Average gas: {:.1f} gwei".format(market['avg_gas_price']))
    
    return results_file, transaction_file

if __name__ == "__main__":
    main() 