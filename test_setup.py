"""
Setup validation script for Uniswap V3 Bot
Run this script to test your configuration before starting the bot
"""

import sys
import logging
from web3 import Web3

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test that all configuration is properly set"""
    logger.info("Testing configuration...")
    
    try:
        from config.settings import settings
        settings.validate()
        logger.info("✓ Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False

def test_web3_connection():
    """Test Web3 connection to Goerli"""
    logger.info("Testing Web3 connection...")
    
    try:
        from config.settings import settings
        web3 = Web3(Web3.HTTPProvider(settings.INFURA_GOERLI_RPC_URL))
        
        if not web3.is_connected():
            logger.error("✗ Failed to connect to Web3")
            return False
        
        block_number = web3.eth.block_number
        logger.info(f"✓ Connected to Web3. Latest block: {block_number}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Web3 connection failed: {e}")
        return False

def test_account_access():
    """Test that we can access the account"""
    logger.info("Testing account access...")
    
    try:
        from config.settings import settings
        from eth_account import Account
        
        account = Account.from_key(settings.PRIVATE_KEY)
        
        if account.address.lower() != settings.PUBLIC_ADDRESS.lower():
            logger.error(f"✗ Private key doesn't match public address")
            logger.error(f"  Private key address: {account.address}")
            logger.error(f"  Configured address: {settings.PUBLIC_ADDRESS}")
            return False
        
        logger.info(f"✓ Account access verified: {account.address}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Account access failed: {e}")
        return False

def test_balances():
    """Test account balances"""
    logger.info("Testing account balances...")
    
    try:
        from config.settings import settings
        from position_manager import PositionManager
        
        web3 = Web3(Web3.HTTPProvider(settings.INFURA_GOERLI_RPC_URL))
        position_manager = PositionManager(web3)
        
        eth_balance, usdc_balance = position_manager.get_token_balances()
        
        logger.info(f"✓ ETH balance: {eth_balance:.6f}")
        logger.info(f"✓ USDC balance: {usdc_balance:.2f}")
        
        if eth_balance < settings.MIN_ETH_BALANCE:
            logger.warning(f"⚠ ETH balance is low (< {settings.MIN_ETH_BALANCE})")
        
        if eth_balance < 0.001:
            logger.warning(f"⚠ Very low ETH balance - you may need more for gas fees")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Balance check failed: {e}")
        return False

def test_price_oracle():
    """Test price oracle functionality"""
    logger.info("Testing price oracle...")
    
    try:
        from config.settings import settings
        from price_oracle import PriceOracle
        
        web3 = Web3(Web3.HTTPProvider(settings.INFURA_GOERLI_RPC_URL))
        oracle = PriceOracle(web3)
        
        price = oracle.get_current_price()
        
        if price is None:
            logger.error("✗ Failed to get current price")
            return False
        
        logger.info(f"✓ Current ETH/USDC price: {price:.6f}")
        
        # Test pool info
        pool_info = oracle.get_pool_info()
        logger.info(f"✓ Pool info retrieved: {pool_info.get('pool_address', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Price oracle test failed: {e}")
        return False

def test_strategy():
    """Test strategy initialization"""
    logger.info("Testing strategy...")
    
    try:
        from strategies.bollinger import BollingerBandsStrategy
        from config.settings import settings
        
        strategy = BollingerBandsStrategy(
            window=settings.BOLLINGER_WINDOW,
            std_dev=settings.BOLLINGER_STD_DEV
        )
        
        # Add some test prices
        test_prices = [2000, 2010, 2005, 1995, 2020]
        for price in test_prices:
            strategy.add_price(price)
        
        stats = strategy.get_statistics()
        logger.info(f"✓ Strategy initialized. Stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Strategy test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🤖 Uniswap V3 Bot Setup Validation")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Web3 Connection", test_web3_connection),
        ("Account Access", test_account_access),
        ("Account Balances", test_balances),
        ("Price Oracle", test_price_oracle),
        ("Strategy", test_strategy),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test '{test_name}' failed")
        except Exception as e:
            logger.error(f"Test '{test_name}' error: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Setup Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup looks good.")
        logger.info("You can now run the bot with: python main.py")
    else:
        logger.error("❌ Some tests failed. Please fix the issues before running the bot.")
        sys.exit(1)

if __name__ == "__main__":
    main() 