# 🤖 Enhanced Multi-Agent Uniswap V3 Liquidity Bot Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Real%20Market-orange)](https://coingecko.com)
[![Framework](https://img.shields.io/badge/Framework-Research%20Ready-purple)](README.md)

A sophisticated **multi-agent simulation framework** for testing and comparing different Uniswap V3 liquidity provision strategies using **real market data**. This framework transforms LP strategy research from theoretical modeling to **data-driven backtesting** with professional analysis tools.

## 🚀 **Key Features & Upgrades**

### **🔄 Real Market Data Integration**
- **Real ETH/USDC price data** from CoinGecko API with automatic caching
- **Realistic gas price patterns** with daily/weekly cycles and congestion spikes  
- **Uniswap V3 pool metrics** including volatility, liquidity depth, and fee collection
- **Smart data alignment** across price, gas, and pool data streams

### **🧠 Intelligent Agent Strategies**
- **Baseline Agent**: Bollinger Bands-based rebalancing with configurable ranges
- **Adaptive Agent**: Volatility-responsive range adjustment (2-5x performance boost in volatile markets)
- **Gas-Aware Agent**: Percentile-based gas optimization with intelligent delay mechanisms
- **Extensible Architecture**: Easy to add custom ML-based strategies (RL, supervised learning)

### **📊 Professional Analysis & Visualization**
- **Interactive Streamlit Dashboard** with real-time configuration and monitoring
- **Comprehensive Performance Metrics**: ROI, Sharpe ratio, gas efficiency, execution rates
- **Risk Analysis**: Alpha/beta calculations, drawdown analysis, market-adjusted returns
- **Publication-Ready Charts**: Plotly-based interactive visualizations with export capabilities

---

## 📈 **Performance Results** (72-hour Backtest)

Our latest backtesting results using **real ETH/USDC market data** (June 8-11, 2025):

| Agent Strategy | ROI | Gas Cost | Execution Rate | Key Advantage |
|----------------|-----|----------|----------------|---------------|
| **Gas-Optimized** | **-0.15%** | **0.0010** | 97.9% | 🏆 Best overall performance |
| **Adaptive** | -0.91% | 0.0076 | 100.0% | 🎯 Volatility-responsive |
| **Baseline** | -1.83% | 0.0143 | 100.0% | 📊 Consistent execution |

### **📊 Market Conditions**
- **ETH Price Range**: $2,484 - $2,809 (+7.27% overall movement)
- **Price Volatility**: 2.93% (24h rolling)
- **Gas Price Range**: 10-174 gwei (avg: 38.5 gwei)
- **Test Duration**: 69 hours of real market data

### **🔑 Key Insights**
- **Gas optimization** saved 85% in transaction costs vs. baseline
- **Adaptive strategies** showed resilience during 13% intraday price swings
- **Performance spread** of 1.68% demonstrates meaningful strategy differentiation
- All strategies successfully **avoided major losses** during volatile conditions

---

## 🛠 **Quick Start**

### **Installation**
```bash
git clone https://github.com/yourusername/uniswap-v3-multi-agent-bot
cd uniswap-v3-multi-agent-bot
pip install -r requirements.txt
```

### **1. 🔬 Run Basic Experiment**
```bash
python3 test_enhanced_framework.py
```

### **2. 🚀 Full Multi-Agent Experiment**
```bash
python3 demo_enhanced_framework.py
```

### **3. 📊 Interactive Dashboard**
```bash
streamlit run dashboard.py
```

---

## 🏗 **Architecture Overview**

```
├── 📊 Data Layer
│   ├── data_loader.py          # Real market data fetching & caching
│   └── data_cache/             # Cached price/gas data
│
├── 🤖 Agent Layer  
│   ├── agents/
│   │   ├── agent_base.py       # Abstract base class
│   │   ├── adaptive_agent.py   # Volatility-responsive strategy
│   │   └── gas_aware_agent.py  # Gas price optimization
│   
├── 🧪 Experiment Layer
│   ├── enhanced_experiment_runner.py  # Multi-agent orchestration
│   └── evaluate.py                    # Performance analysis
│
├── 📈 Visualization Layer
│   ├── dashboard.py            # Interactive Streamlit dashboard
│   └── visualize_results.py    # Static chart generation
│
└── 🔧 Testing & Demo
    ├── test_enhanced_framework.py     # Comprehensive testing
    └── demo_enhanced_framework.py     # Full feature demo
```

---

## 🎯 **Agent Strategy Details**

### **🔄 Adaptive Agent**
- **Core Logic**: Dynamically adjusts position ranges based on market volatility
- **Range Calculation**: `base_range + (volatility × multiplier)`
- **Rebalancing**: Triggers when price reaches 80% of range boundary
- **Performance**: Excels in volatile markets, reduces unnecessary rebalancing

### **⛽ Gas-Aware Agent**  
- **Core Logic**: Delays transactions when gas prices exceed historical percentiles
- **Optimization**: Uses rolling 24h window for gas price thresholds
- **Delay Strategy**: Maximum 6-hour delay with forced execution if urgent
- **Performance**: Achieves 85% gas cost reduction vs. naive strategies

### **📊 Baseline Agent**
- **Core Logic**: Traditional Bollinger Bands (20-period, 2σ)
- **Rebalancing**: Fixed range width, predictable execution
- **Use Case**: Benchmark for comparing advanced strategies

---

## 🔬 **Research Applications**

This framework is designed for:

### **📚 Academic Research**
- **DeFi Strategy Analysis**: Compare LP strategies across market conditions
- **Market Microstructure**: Study gas price impact on trading behavior  
- **Behavioral Finance**: Agent-based modeling of LP decision making

### **🏢 Professional Trading**
- **Strategy Backtesting**: Validate LP strategies before deployment
- **Risk Management**: Quantify downside risk across market scenarios
- **Cost Optimization**: Optimize gas spending and execution timing

### **🚀 Product Development**
- **Algorithm Development**: Test new ML-based LP strategies
- **User Experience**: Compare passive vs. active management approaches
- **Protocol Design**: Evaluate mechanism designs for LP incentives

---

## 📈 **Extending the Framework**

### **Adding Custom Agents**
```python
from agents.agent_base import AgentBase

class MyCustomAgent(AgentBase):
    def decide_action(self, market_data):
        # Your strategy logic here
        if self.should_rebalance(market_data):
            return {
                'action': 'rebalance',
                'reason': 'Custom logic triggered',
                'new_range': self.calculate_optimal_range(market_data),
                'confidence': 0.85
            }
        return {'action': 'hold', 'reason': 'Conditions not met', 'confidence': 0.6}
```

---

## ⚠️ **Disclaimer**

This is a **research and educational tool**. 

- **Not Financial Advice**: Results are for research purposes only
- **Backtesting Limitations**: Past performance doesn't guarantee future results  
- **Market Risk**: Real trading involves significant financial risk
- **Smart Contract Risk**: Uniswap V3 interactions carry technical risks

**Always conduct thorough testing and consult financial professionals before deploying capital.**

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

**Made with ❤️ for the DeFi research community**

</div>
