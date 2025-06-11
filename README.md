# Uniswap V3 Liquidity Management Bot

An automated Python bot that manages liquidity positions on Uniswap V3 using a Bollinger Bands strategy on the Goerli testnet.

## Features

- 🔄 **Automated Liquidity Management**: Automatically adds and removes liquidity based on market conditions
- 📊 **Bollinger Bands Strategy**: Uses statistical analysis to determine optimal price ranges
- 🌐 **Goerli Testnet**: Safe testing environment without real money at risk
- 📈 **Real-time Price Monitoring**: Fetches current ETH/USDC prices from Uniswap V3 pool
- 📝 **Comprehensive Logging**: Detailed logs and CSV export of all actions
- ⚡ **Gas Optimization**: Built-in safety checks for gas prices and balances
- 🔐 **Secure**: Private keys loaded from environment variables

## Project Structure

```
├── main.py                 # Main application loop
├── price_oracle.py         # Price fetching from Uniswap V3 pool
├── position_manager.py     # Liquidity position management
├── strategies/
│   ├── __init__.py
│   └── bollinger.py       # Bollinger Bands strategy implementation
├── config/
│   ├── __init__.py
│   ├── contracts.py       # Contract addresses and ABIs
│   └── settings.py        # Configuration management
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Prerequisites

- Python 3.8+
- Goerli testnet ETH and USDC
- Infura account (or other Ethereum RPC provider)
- MetaMask or other wallet with private key access

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd uniswap-v3-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

4. **Configure environment variables:**
   Edit `.env` file with your credentials:
   ```env
   PRIVATE_KEY=0x...your_private_key...
   PUBLIC_ADDRESS=0x...your_wallet_address...
   INFURA_GOERLI_RPC_URL=https://goerli.infura.io/v3/your_project_id
   
   # Optional configurations
   CHECK_INTERVAL_MINUTES=5
   BOLLINGER_WINDOW=20
   BOLLINGER_STD_DEV=2.0
   MIN_POSITION_SIZE_ETH=0.001
   MAX_POSITION_SIZE_ETH=0.01
   ```

## Getting Goerli Testnet Tokens

1. **Get Goerli ETH:**
   - Use a Goerli faucet like https://goerlifaucet.com/
   - Or https://faucet.paradigm.xyz/

2. **Get Goerli USDC:**
   - Use Uniswap V3 on Goerli to swap some ETH for USDC
   - Or use the USDC faucet if available

3. **Wrap ETH to WETH (if needed):**
   - The bot works with WETH, not native ETH
   - Use the WETH contract to wrap your ETH

## Usage

### Basic Usage

```bash
python main.py
```

The bot will:
1. Connect to the Goerli network
2. Initialize the price oracle and position manager
3. Start monitoring ETH/USDC prices every 5 minutes (configurable)
4. Apply Bollinger Bands strategy to determine if rebalancing is needed
5. Add/remove liquidity positions as necessary
6. Log all actions to CSV files

### Configuration Options

All configuration is done through environment variables:

- `CHECK_INTERVAL_MINUTES`: How often to check prices (default: 5)
- `BOLLINGER_WINDOW`: Number of price periods for Bollinger Bands (default: 20)
- `BOLLINGER_STD_DEV`: Standard deviations for bands (default: 2.0)
- `MIN_POSITION_SIZE_ETH`: Minimum ETH to use per position (default: 0.001)
- `MAX_POSITION_SIZE_ETH`: Maximum ETH to use per position (default: 0.01)
- `MAX_GAS_PRICE_GWEI`: Maximum gas price before skipping transactions (default: 50)
- `MIN_ETH_BALANCE`: Minimum ETH balance to maintain (default: 0.01)

## Strategy Details

### Bollinger Bands

The bot uses Bollinger Bands to determine optimal liquidity ranges:

1. **Price Collection**: Collects ETH/USDC prices every check interval
2. **Band Calculation**: Calculates moving average ± 2 standard deviations over the last 20 prices
3. **Range Setting**: Uses the bands as the price range for liquidity positions
4. **Rebalancing Triggers**:
   - Current price moves outside the existing position range
   - The optimal range deviates significantly from the current position range
   - No active position exists

### Safety Features

- **Gas Price Monitoring**: Skips transactions when gas prices are too high
- **Balance Checks**: Ensures sufficient ETH balance before transactions
- **Slippage Protection**: Uses minimum amount parameters in transactions
- **Error Handling**: Comprehensive error handling and logging

## Monitoring and Logs

### Log Files

- `bot.log`: Detailed application logs
- `bot_actions_YYYYMMDD_HHMMSS.csv`: CSV file with all bot actions

### CSV Log Columns

- `timestamp`: When the action occurred
- `action`: Type of action (price_update, mint_position, remove_liquidity, etc.)
- `current_price`: ETH/USDC price at the time
- `lower_range`/`upper_range`: Position price range
- `position_id`: Uniswap V3 NFT token ID
- `eth_amount`/`usdc_amount`: Token amounts involved
- `gas_used`: Gas consumed by transaction
- `tx_hash`: Transaction hash
- `status`: Success/failure status
- `notes`: Additional information

## Important Notes

### ⚠️ Testnet Only

This bot is designed for **Goerli testnet only**. Do not use on mainnet without thorough testing and additional safety measures.

### 🔐 Security

- Never commit your `.env` file to version control
- Use a dedicated wallet for bot operations
- Keep only small amounts of testnet tokens in the bot wallet
- Review all transactions before running in any production environment

### 📊 Performance

- The bot may take 20+ price readings before the Bollinger Bands strategy becomes effective
- Initial positions may be based on simple price ranges until sufficient history is built
- Performance depends on market volatility and gas prices

### 🔧 Customization

The bot is modular and can be extended:

- Add new strategies in the `strategies/` directory
- Modify position sizing logic in `BollingerBandsStrategy.get_position_size()`
- Add new safety checks in `UniswapV3Bot.check_safety_conditions()`
- Implement different rebalancing triggers

## Troubleshooting

### Common Issues

1. **"Configuration error: Missing required environment variables"**
   - Ensure all required variables are set in `.env` file
   - Check that private key and address are correctly formatted

2. **"Failed to connect to Web3"**
   - Verify your Infura URL is correct
   - Check your internet connection
   - Ensure the Infura project has Goerli access enabled

3. **"Insufficient price history"**
   - Wait for the bot to collect more price data
   - The strategy needs at least 20 price points to work effectively

4. **Transaction failures**
   - Check that you have sufficient Goerli ETH for gas
   - Verify token balances and allowances
   - Monitor gas prices during high network activity

### Getting Help

- Check the `bot.log` file for detailed error messages
- Review the CSV logs to understand bot behavior
- Ensure all prerequisites are met
- Test with small amounts first

## Disclaimer

This software is for educational and testing purposes only. Use at your own risk. The authors are not responsible for any losses incurred through the use of this bot. Always test thoroughly on testnets before considering any mainnet usage.

## License

MIT License - see LICENSE file for details. 