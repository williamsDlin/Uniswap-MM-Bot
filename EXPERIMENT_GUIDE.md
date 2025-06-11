# Multi-Agent Bot Experiment Guide 🤖

Welcome to the comprehensive experiment guide for the Multi-Agent Uniswap V3 Bot Framework! This guide will walk you through running experiments to compare and analyze different bot strategies.

## 🚀 Quick Start

### Option 1: Interactive Experiment Runner (Recommended)
```bash
python run_experiments.py
```

### Option 2: Streamlit Dashboard
```bash
streamlit run experiments/dashboard.py
```

### Option 3: Manual Experiments
```bash
python experiments/runner.py
```

## 📋 Prerequisites

### 1. Dependencies
Ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

Required packages:
- `pandas>=2.1.3` - Data manipulation
- `numpy>=1.25.2` - Numerical computing  
- `plotly>=5.17.0` - Interactive visualizations
- `streamlit>=1.28.0` - Web dashboard
- `torch>=2.1.0` - ML models
- `scikit-learn>=1.3.0` - ML utilities
- `statsmodels>=0.14.0` - Time series analysis

### 2. Project Structure
Your workspace should have this structure:
```
📦 Uniswap MM Bot/
├── 🤖 Multi-Agent Framework
│   ├── ml/                    # ML agents
│   ├── strategies/            # Trading strategies  
│   ├── experiments/           # Experiment tools
│   ├── config/               # Configuration
│   └── evaluation/           # Performance evaluation
├── 📊 Experiment Tools
│   ├── run_experiments.py    # Main experiment runner
│   └── EXPERIMENT_GUIDE.md   # This guide
└── 📁 Generated Data
    ├── charts/               # HTML visualizations
    ├── transaction_data_*.csv # Transaction logs
    └── experiment_results_*.json # Experiment results
```

## 🔬 Experiment Types

### 1. Basic Performance Comparison
**Goal**: Compare all 4 bot types in a standard market scenario

**Bots Tested**:
- 🧠 **Adaptive Bollinger**: Uses LinUCB bandit learning for dynamic parameters
- 🎮 **RL Rebalancer**: Uses PPO reinforcement learning for rebalancing decisions  
- ⛽ **Gas Aware**: Uses SARIMA time series forecasting for gas optimization
- 📊 **Baseline**: Simple static Bollinger Bands strategy

**Configuration**:
- Duration: 1000 steps (≈3.5 days)
- Market: ETH/USDC with realistic volatility
- Gas: Daily patterns with congestion spikes

**Expected Results**:
- Adaptive Bollinger: +2-5% ROI improvement
- RL Rebalancer: Better timing, fewer unnecessary rebalances
- Gas Aware: 15-30% gas cost reduction
- Baseline: Consistent but suboptimal performance

### 2. Parameter Sensitivity Analysis
**Goal**: Test how bots perform under different time horizons

**Test Scenarios**:
- Short-term: 500 steps (≈1.7 days)
- Medium-term: 1000 steps (≈3.5 days) 
- Long-term: 2000 steps (≈7 days)

**Key Metrics**:
- ROI consistency across timeframes
- Rebalance frequency adaptation
- Gas efficiency trends

### 3. Market Condition Testing
**Goal**: Evaluate robustness under different market conditions

**Market Scenarios**:
- Low volatility (1% daily)
- Normal volatility (2% daily)
- High volatility (5% daily)
- Trending markets (bull/bear)
- Sideways markets (ranging)

### 4. Stress Testing
**Goal**: Test bot performance under extreme conditions

**Stress Scenarios**:
- Flash crashes (>20% price drops)
- Gas fee spikes (>200 gwei)
- Network congestion simulation
- MEV sandwich attacks

## 📊 Performance Metrics

### Primary Metrics
1. **ROI (Return on Investment)**: `(final_capital - initial_capital) / initial_capital * 100`
2. **Sharpe Ratio**: Risk-adjusted returns
3. **Gas Efficiency**: `fees_earned / gas_cost_paid`
4. **Position Effectiveness**: `time_in_range / total_time * 100`

### Secondary Metrics
1. **Rebalance Frequency**: Adaptivity indicator
2. **Maximum Drawdown**: Risk measurement
3. **Win Rate**: Percentage of profitable periods
4. **Volatility**: Strategy stability

### ML-Specific Metrics
1. **Learning Rate**: How quickly agents improve
2. **Exploration vs Exploitation**: Balance in decision making
3. **Confidence Scores**: Agent certainty levels
4. **Prediction Accuracy**: For gas price forecasting

## 🎯 Running Experiments

### Method 1: Interactive Runner (Recommended)

1. **Start the runner**:
   ```bash
   python run_experiments.py
   ```

2. **Choose experiment type**:
   ```
   🔧 Choose your experiment:
     1. Full Experiment Suite (recommended)
     2. Quick Demo (fast)
     3. Just Launch Dashboard
     4. Parameter Analysis Only
     5. Generate Visualizations Only
   ```

3. **Monitor progress**:
   - Real-time step completion
   - Bot performance summaries
   - Error handling and recovery

4. **Review results**:
   - Automatic file generation
   - Performance summaries
   - Visualization creation

### Method 2: Streamlit Dashboard

1. **Launch dashboard**:
   ```bash
   streamlit run experiments/dashboard.py
   ```

2. **Configure experiment**:
   - Select number of steps (100-2000)
   - Choose bot types to compare
   - Set market parameters

3. **Run experiment**:
   - Click "Start Experiment"
   - Monitor real-time results
   - View live performance metrics

4. **Analyze results**:
   - Interactive charts
   - Performance comparisons
   - Export capabilities

### Method 3: Command Line

1. **Basic experiment**:
   ```python
   from experiments.runner import ExperimentRunner
   
   runner = ExperimentRunner()
   runner.add_bot("adaptive", "adaptive_bollinger")
   runner.add_bot("baseline", "baseline")
   
   results = runner.run_experiment(1000)
   runner.save_results()
   runner.save_transaction_data()
   ```

2. **Custom visualization**:
   ```python
   from experiments.visualizer import BotPerformanceVisualizer
   
   viz = BotPerformanceVisualizer()
   viz.load_transaction_data("transaction_data_20241201_123456.csv")
   viz.load_results_data("experiment_results_20241201_123456.json")
   viz.save_all_charts()
   ```

## 📈 Understanding Results

### Generated Files

1. **experiment_results_TIMESTAMP.json**:
   - High-level performance summary
   - Bot configurations used
   - Market conditions
   - Key metrics comparison

2. **transaction_data_TIMESTAMP.csv**:
   - Step-by-step transaction log
   - Price, gas, and action data
   - Capital tracking
   - Detailed bot decisions

3. **parameter_analysis_TIMESTAMP.json**:
   - Sensitivity analysis results
   - Performance across different configurations
   - Optimization recommendations

### Visualization Outputs

1. **charts/roi_comparison.html**:
   - ROI over time for all bots
   - Interactive plotly chart
   - Hover details and zoom

2. **charts/dashboard.html**:
   - Comprehensive 4-panel view
   - ROI, capital, rebalances, efficiency
   - Multi-metric comparison

3. **charts/summary.html**:
   - Performance summary table
   - Ranked bot comparison
   - Key statistics

### Reading Performance Data

**ROI Interpretation**:
- Positive ROI: Bot is profitable
- Higher ROI: Better performance
- Consistent ROI: More reliable strategy

**Rebalance Frequency**:
- Too high: Excessive gas costs
- Too low: Missing opportunities
- Optimal: Balanced responsiveness

**Gas Efficiency**:
- Ratio > 1: Profitable after gas costs
- Higher ratio: More efficient execution
- Gas-aware bots should excel here

## 🔧 Advanced Configuration

### Custom Market Scenarios

```python
# Create custom market data
from experiments.runner import MarketDataGenerator

generator = MarketDataGenerator(
    initial_price=2000.0,
    volatility=0.03,  # 3% volatility
    trend=0.0001      # Slight upward trend
)

market_data = generator.generate_price_series(
    num_steps=1000,
    crash_probability=0.01,  # 1% chance of crash per step
    gas_spikes=True          # Include gas spikes
)
```

### Bot Parameter Tuning

```python
# Customize bot parameters
adaptive_bot = BotSimulator("adaptive", "adaptive_bollinger")
adaptive_bot.strategy.update_parameters(
    window_range=(10, 30),     # Bollinger window range
    std_dev_range=(1.5, 3.0),  # Standard deviation range
    learning_rate=0.1          # Bandit learning rate
)
```

### Experiment Batching

```python
# Run multiple experiments
results_batch = []

for volatility in [0.01, 0.02, 0.03, 0.04, 0.05]:
    runner = ExperimentRunner()
    runner.add_bot("adaptive", "adaptive_bollinger")
    runner.add_bot("baseline", "baseline")
    
    # Set market volatility
    runner.market_volatility = volatility
    
    results = runner.run_experiment(1000)
    results['volatility'] = volatility
    results_batch.append(results)

# Analyze batch results
for result in results_batch:
    vol = result['volatility']
    adaptive_roi = result['bot_performance']['adaptive']['final_roi']
    print(f"Volatility {vol:.1%}: Adaptive ROI = {adaptive_roi:.2f}%")
```

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: No module named 'experiments'**
   ```bash
   # Make sure you're in the correct directory
   cd "path/to/Uniswap MM Bot"
   python run_experiments.py
   ```

2. **Streamlit command not found**
   ```bash
   pip install streamlit
   # or
   python -m streamlit run experiments/dashboard.py
   ```

3. **Empty visualizations**
   - Check if data files exist
   - Ensure experiments completed successfully
   - Verify file permissions

4. **Memory errors with large experiments**
   ```python
   # Reduce experiment size
   results = runner.run_experiment(500)  # Instead of 2000
   ```

5. **Slow performance**
   ```python
   # Use parallel execution (if available)
   results = runner.run_experiment(1000, parallel=True)
   ```

### Performance Tips

1. **Start small**: Begin with 100-500 steps
2. **Monitor memory**: Large experiments can use significant RAM
3. **Save frequently**: Enable auto-save for long experiments
4. **Use SSD**: Faster disk I/O improves performance
5. **Close other apps**: Free up system resources

## 📚 Next Steps

### Analyzing Results
1. **Compare ROI trends**: Which bot performs best over time?
2. **Examine rebalance patterns**: When do bots make decisions?
3. **Study gas efficiency**: How much do transaction costs matter?
4. **Look for edge cases**: When do strategies fail?

### Optimization
1. **Parameter tuning**: Adjust bot configurations
2. **Strategy hybridization**: Combine successful elements
3. **Risk management**: Add stop-loss mechanisms
4. **Market adaptation**: Develop regime-aware strategies

### Real Deployment
1. **Paper trading**: Test with live market data
2. **Risk assessment**: Evaluate maximum potential loss
3. **Monitoring setup**: Real-time performance tracking
4. **Gradual scaling**: Start with small position sizes

## 🎓 Learning Objectives

After completing experiments, you should understand:

1. **Strategy Comparison**: Relative strengths/weaknesses of each approach
2. **Market Sensitivity**: How market conditions affect performance
3. **Parameter Impact**: Which settings matter most
4. **Gas Optimization**: The importance of transaction cost management
5. **ML Integration**: How AI agents can improve traditional strategies
6. **Risk Management**: The role of volatility and drawdowns
7. **Performance Measurement**: Meaningful metrics for strategy evaluation

## 💡 Pro Tips

1. **Run multiple seeds**: Use different random seeds for robustness
2. **Time diversification**: Test across different market periods
3. **Correlation analysis**: Understand when strategies agree/disagree
4. **Scenario planning**: Prepare for extreme market events
5. **Continuous learning**: Regularly update ML models with new data
6. **Documentation**: Keep detailed notes on experiment configurations
7. **Version control**: Track changes to strategies and parameters

## 🔗 Additional Resources

- [Multi-Agent README](README_MULTI_AGENT.md) - Architecture overview
- [Bot Evaluation Framework](evaluation/evaluator.py) - Advanced metrics
- [ML Agent Documentation](ml/) - Deep dive into AI components
- [Strategy Development Guide](strategies/) - Creating custom strategies

---

🚀 **Ready to start experimenting?** Run `python run_experiments.py` and choose option 1 for the full experience!

For questions or issues, check the troubleshooting section or examine the generated log files for detailed error information. 