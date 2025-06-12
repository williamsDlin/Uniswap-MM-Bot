#!/usr/bin/env python3
"""
Simple Multi-Agent Bot Demo - Testing Basic Functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("Simple Multi-Agent Bot Demo Starting...")
print("=" * 50)

# Generate synthetic market data
print("Generating market data...")
np.random.seed(42)

# Market simulation parameters
num_steps = 100
initial_price = 2000.0
current_price = initial_price

prices = []
gas_prices = []

for step in range(num_steps):
    # Price random walk
    change = np.random.normal(0, 0.01)
    current_price *= (1 + change)
    prices.append(current_price)
    
    # Gas price pattern
    hour = (step * 5 / 60) % 24
    base_gas = 20 + 30 * np.sin(hour * np.pi / 12)
    gas_price = max(10, base_gas + np.random.normal(0, 5))
    gas_prices.append(gas_price)

print("Market data generated: {} steps".format(num_steps))
print("Price range: ${:.2f} - ${:.2f}".format(min(prices), max(prices)))
print("Gas range: {:.1f} - {:.1f} gwei".format(min(gas_prices), max(gas_prices)))

# Simulate different bot strategies
class SimpleBot:
    def __init__(self, name, strategy_type):
        self.name = name
        self.strategy_type = strategy_type
        self.capital = 1.0
        self.transactions = []
        self.rebalance_count = 0
        
    def should_rebalance(self, price, price_history):
        if len(price_history) < 20:
            return False
            
        if self.strategy_type == "baseline":
            # Simple Bollinger Bands
            window = price_history[-20:]
            mean_price = np.mean(window)
            std_price = np.std(window)
            
            if price < mean_price - 2 * std_price or price > mean_price + 2 * std_price:
                return True
                
        elif self.strategy_type == "adaptive":
            # More aggressive rebalancing
            window = price_history[-15:]
            mean_price = np.mean(window)
            std_price = np.std(window)
            
            if price < mean_price - 1.5 * std_price or price > mean_price + 1.5 * std_price:
                return True
                
        elif self.strategy_type == "gas_aware":
            # Only rebalance when gas is low
            if len(price_history) >= 20:
                window = price_history[-20:]
                mean_price = np.mean(window)
                std_price = np.std(window)
                
                if price < mean_price - 2 * std_price or price > mean_price + 2 * std_price:
                    return True
                    
        return False
    
    def step(self, price, gas_price, price_history):
        action = "hold"
        fees = 0.0
        gas_cost = 0.0
        
        if self.should_rebalance(price, price_history):
            # Different gas sensitivity
            if self.strategy_type == "gas_aware" and gas_price > 40:
                action = "gas_delayed"
            else:
                action = "rebalance"
                self.rebalance_count += 1
                fees = 0.0003  # 0.03% fee
                gas_cost = 0.01 * (gas_price / 30.0)
        
        # Update capital
        net_change = fees - gas_cost
        self.capital += net_change
        
        # Record transaction
        transaction = {
            'price': price,
            'gas_price': gas_price,
            'action': action,
            'capital': self.capital,
            'fees': fees,
            'gas_cost': gas_cost
        }
        self.transactions.append(transaction)
        
        return transaction

# Create bots
bots = [
    SimpleBot("Baseline", "baseline"),
    SimpleBot("Adaptive", "adaptive"), 
    SimpleBot("Gas-Aware", "gas_aware"),
    SimpleBot("Aggressive", "adaptive")  # Another adaptive variant
]

print("\nRunning simulation...")

# Run simulation
for step in range(num_steps):
    current_price = prices[step]
    current_gas = gas_prices[step]
    price_history = prices[:step+1]
    
    for bot in bots:
        bot.step(current_price, current_gas, price_history)
    
    if step % 20 == 0:
        print("Step {}/{}".format(step, num_steps))

print("\nSimulation completed!")

# Calculate results
print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)

for bot in bots:
    roi = ((bot.capital - 1.0) / 1.0) * 100
    total_fees = sum(t['fees'] for t in bot.transactions)
    total_gas = sum(t['gas_cost'] for t in bot.transactions)
    net_profit = total_fees - total_gas
    
    print("{:12} | ROI: {:6.2f}% | Rebalances: {:3d} | Net: {:8.6f}".format(
        bot.name, roi, bot.rebalance_count, net_profit))

# Market summary
price_change = ((prices[-1] - prices[0]) / prices[0]) * 100
avg_gas = np.mean(gas_prices)

print("\nMarket Summary:")
print("Price change: {:.2f}%".format(price_change))
print("Average gas: {:.1f} gwei".format(avg_gas))
print("Volatility: {:.2f}%".format(np.std(np.diff(prices) / prices[:-1]) * 100))

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create transaction dataframe
all_transactions = []
for i, bot in enumerate(bots):
    for j, transaction in enumerate(bot.transactions):
        transaction['step'] = j
        transaction['bot_id'] = bot.name
        transaction['bot_type'] = bot.strategy_type
        all_transactions.append(transaction)

df = pd.DataFrame(all_transactions)
transaction_file = "simple_demo_transactions_{}.csv".format(timestamp)
df.to_csv(transaction_file, index=False)

print("\nData saved to: {}".format(transaction_file))
print("Demo completed successfully!")

# Show a few sample transactions
print("\nSample transactions (last 5 for each bot):")
for bot in bots:
    print("\n{}:".format(bot.name))
    for transaction in bot.transactions[-3:]:
        print("  Price: ${:.2f}, Gas: {:.1f}, Action: {}".format(
            transaction['price'], transaction['gas_price'], transaction['action']))

print("\n✅ Simple demo completed successfully!") 