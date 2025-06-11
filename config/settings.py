"""
Bot settings and configuration management
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Network Configuration
    PRIVATE_KEY = os.getenv('PRIVATE_KEY')
    PUBLIC_ADDRESS = os.getenv('PUBLIC_ADDRESS')
    INFURA_GOERLI_RPC_URL = os.getenv('INFURA_GOERLI_RPC_URL')
    ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
    
    # Bot Configuration
    CHECK_INTERVAL_MINUTES = int(os.getenv('CHECK_INTERVAL_MINUTES', 5))
    BOLLINGER_WINDOW = int(os.getenv('BOLLINGER_WINDOW', 20))
    BOLLINGER_STD_DEV = float(os.getenv('BOLLINGER_STD_DEV', 2.0))
    MIN_POSITION_SIZE_ETH = float(os.getenv('MIN_POSITION_SIZE_ETH', 0.001))
    MAX_POSITION_SIZE_ETH = float(os.getenv('MAX_POSITION_SIZE_ETH', 0.01))
    
    # Gas Configuration
    MAX_GAS_PRICE_GWEI = int(os.getenv('MAX_GAS_PRICE_GWEI', 50))
    GAS_LIMIT_BUFFER = float(os.getenv('GAS_LIMIT_BUFFER', 1.2))
    
    # Safety Configuration
    MAX_SLIPPAGE_PERCENT = float(os.getenv('MAX_SLIPPAGE_PERCENT', 1.0))
    MIN_ETH_BALANCE = float(os.getenv('MIN_ETH_BALANCE', 0.01))
    
    @classmethod
    def validate(cls):
        """Validate that all required settings are present"""
        required_settings = [
            'PRIVATE_KEY',
            'PUBLIC_ADDRESS', 
            'INFURA_GOERLI_RPC_URL'
        ]
        
        missing_settings = []
        for setting in required_settings:
            if not getattr(cls, setting):
                missing_settings.append(setting)
        
        if missing_settings:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_settings)}")
        
        # Validate addresses
        if not cls.PUBLIC_ADDRESS.startswith('0x') or len(cls.PUBLIC_ADDRESS) != 42:
            raise ValueError("Invalid PUBLIC_ADDRESS format")
            
        if not cls.PRIVATE_KEY.startswith('0x') or len(cls.PRIVATE_KEY) != 66:
            raise ValueError("Invalid PRIVATE_KEY format")

# Create global settings instance
settings = Settings() 