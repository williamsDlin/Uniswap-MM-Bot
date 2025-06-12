#!/usr/bin/env python3
"""
Visualization of Multi-Agent Bot Experiment Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Find the latest transaction file
print("Looking for experiment data files...")
transaction_files = glob.glob("simple_demo_transactions_*.csv")

if not transaction_files:
    print("No transaction files found. Run simple_demo.py first.")
    exit(1)

# Use the latest file
latest_file = sorted(transaction_files)[-1]
print("Using file: {}".format(latest_file))

# Load data
df = pd.DataFrame(pd.read_csv(latest_file))
print("Loaded {} transactions".format(len(df)))

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Multi-Agent Bot Performance Analysis', fontsize=16, fontweight='bold')

# 1. Capital Over Time
ax1 = axes[0, 0]
for bot_type in df['bot_id'].unique():
    bot_data = df[df['bot_id'] == bot_type]
    ax1.plot(bot_data['step'], bot_data['capital'], label=bot_type, linewidth=2)

ax1.set_title('Capital Over Time')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Capital')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. ROI Comparison
ax2 = axes[0, 1]
bot_summary = []
for bot_type in df['bot_id'].unique():
    bot_data = df[df['bot_id'] == bot_type]
    final_capital = bot_data['capital'].iloc[-1]
    roi = ((final_capital - 1.0) / 1.0) * 100
    bot_summary.append({'bot': bot_type, 'roi': roi})

bot_summary_df = pd.DataFrame(bot_summary)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax2.bar(bot_summary_df['bot'], bot_summary_df['roi'], color=colors)
ax2.set_title('Final ROI Comparison')
ax2.set_xlabel('Bot Type')
ax2.set_ylabel('ROI (%)')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, roi in zip(bars, bot_summary_df['roi']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.5),
             '{:.1f}%'.format(roi), ha='center', va='bottom' if height >= 0 else 'top')

# 3. Market Price and Gas
ax3 = axes[1, 0]
market_data = df[df['bot_id'] == df['bot_id'].unique()[0]]  # Use first bot's data for market
ax3_gas = ax3.twinx()

line1 = ax3.plot(market_data['step'], market_data['price'], 'b-', label='ETH Price', linewidth=2)
line2 = ax3_gas.plot(market_data['step'], market_data['gas_price'], 'r-', label='Gas Price', linewidth=2)

ax3.set_title('Market Conditions')
ax3.set_xlabel('Time Steps')
ax3.set_ylabel('ETH Price ($)', color='b')
ax3_gas.set_ylabel('Gas Price (gwei)', color='r')
ax3.tick_params(axis='y', labelcolor='b')
ax3_gas.tick_params(axis='y', labelcolor='r')

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper right')

# 4. Action Frequency
ax4 = axes[1, 1]
action_summary = []
for bot_type in df['bot_id'].unique():
    bot_data = df[df['bot_id'] == bot_type]
    rebalances = len(bot_data[bot_data['action'] == 'rebalance'])
    holds = len(bot_data[bot_data['action'] == 'hold'])
    delays = len(bot_data[bot_data['action'] == 'gas_delayed'])
    action_summary.append({
        'bot': bot_type,
        'rebalances': rebalances,
        'holds': holds,
        'delays': delays
    })

action_df = pd.DataFrame(action_summary)
width = 0.25
x = np.arange(len(action_df))

ax4.bar(x - width, action_df['rebalances'], width, label='Rebalances', color='#ff7f0e')
ax4.bar(x, action_df['holds'], width, label='Holds', color='#1f77b4')
ax4.bar(x + width, action_df['delays'], width, label='Gas Delays', color='#d62728')

ax4.set_title('Action Frequency')
ax4.set_xlabel('Bot Type')
ax4.set_ylabel('Count')
ax4.set_xticks(x)
ax4.set_xticklabels(action_df['bot'])
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plot_filename = "experiment_analysis_{}.png".format(latest_file.split('_')[-1].replace('.csv', ''))
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print("Visualization saved as: {}".format(plot_filename))

# Display performance summary
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)

for _, row in bot_summary_df.iterrows():
    bot_type = row['bot']
    roi = row['roi']
    
    bot_data = df[df['bot_id'] == bot_type]
    rebalances = len(bot_data[bot_data['action'] == 'rebalance'])
    total_fees = bot_data['fees'].sum()
    total_gas = bot_data['gas_cost'].sum()
    
    print("{:12} | ROI: {:6.2f}% | Rebalances: {:3d} | Fees: {:8.6f} | Gas: {:8.6f}".format(
        bot_type, roi, rebalances, total_fees, total_gas))

# Market analysis
market_data = df[df['bot_id'] == df['bot_id'].unique()[0]]
initial_price = market_data['price'].iloc[0]
final_price = market_data['price'].iloc[-1]
price_change = ((final_price - initial_price) / initial_price) * 100
avg_gas = market_data['gas_price'].mean()
volatility = np.std(np.diff(market_data['price']) / market_data['price'].iloc[:-1]) * 100

print("\nMarket Analysis:")
print("Price change: {:.2f}% (${:.2f} -> ${:.2f})".format(price_change, initial_price, final_price))
print("Average gas: {:.1f} gwei".format(avg_gas))
print("Price volatility: {:.2f}%".format(volatility))

print("\nKey Insights:")

# Find best performing bot
best_bot = bot_summary_df.loc[bot_summary_df['roi'].idxmax()]
worst_bot = bot_summary_df.loc[bot_summary_df['roi'].idxmin()]

print("- Best performer: {} with {:.2f}% ROI".format(best_bot['bot'], best_bot['roi']))
print("- Worst performer: {} with {:.2f}% ROI".format(worst_bot['bot'], worst_bot['roi']))

# Gas awareness analysis
gas_aware_data = df[df['bot_id'] == 'Gas-Aware']
gas_delays = len(gas_aware_data[gas_aware_data['action'] == 'gas_delayed'])
if gas_delays > 0:
    print("- Gas-Aware bot delayed {} transactions due to high gas prices".format(gas_delays))
else:
    print("- Gas-Aware bot had no gas-related delays in this simulation")

# Rebalancing activity
most_active = action_df.loc[action_df['rebalances'].idxmax()]
least_active = action_df.loc[action_df['rebalances'].idxmin()]

print("- Most active rebalancer: {} with {} rebalances".format(most_active['bot'], most_active['rebalances']))
print("- Least active rebalancer: {} with {} rebalances".format(least_active['bot'], least_active['rebalances']))

print("\n✅ Analysis complete! Check {} for visualizations.".format(plot_filename)) 