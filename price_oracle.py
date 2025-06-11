"""
Price Oracle for fetching ETH/USDC prices from Uniswap V3
"""

import math
import logging
from typing import Optional, Tuple
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3Exception

from config.contracts import POOL_ABI, ETH_USDC_POOL, WETH9, USDC

logger = logging.getLogger(__name__)

class PriceOracle:
    def __init__(self, web3: Web3, pool_address: str = ETH_USDC_POOL):
        """
        Initialize Price Oracle
        
        Args:
            web3: Web3 instance
            pool_address: Uniswap V3 pool contract address
        """
        self.web3 = web3
        self.pool_address = pool_address
        self.pool_contract = web3.eth.contract(address=pool_address, abi=POOL_ABI)
        
        # Cache token info
        self.token0 = None
        self.token1 = None
        self.fee = None
        self._initialize_pool_info()
    
    def _initialize_pool_info(self) -> None:
        """Initialize pool information"""
        try:
            self.token0 = self.pool_contract.functions.token0().call()
            self.token1 = self.pool_contract.functions.token1().call()
            self.fee = self.pool_contract.functions.fee().call()
            
            logger.info(f"Pool initialized - Token0: {self.token0}, Token1: {self.token1}, Fee: {self.fee}")
            
            # Determine if this is ETH/USDC or USDC/ETH
            self.is_eth_token0 = self.token0.lower() == WETH9.lower()
            
            if not self.is_eth_token0 and self.token1.lower() != WETH9.lower():
                logger.warning("Pool doesn't contain WETH - price calculations may be incorrect")
                
        except Exception as e:
            logger.error(f"Failed to initialize pool info: {e}")
            raise
    
    def get_current_price(self) -> Optional[float]:
        """
        Get current ETH/USDC price from the pool
        
        Returns:
            Current price as float or None if error
        """
        try:
            # Call slot0() to get current price info
            slot0_result = self.pool_contract.functions.slot0().call()
            sqrt_price_x96 = slot0_result[0]
            tick = slot0_result[1]
            
            # Convert sqrtPriceX96 to actual price
            price = self._sqrt_price_x96_to_price(sqrt_price_x96)
            
            # If ETH is token1, we need to invert the price
            if not self.is_eth_token0:
                price = 1 / price
            
            logger.debug(f"Current price: {price:.6f} USDC/ETH (sqrt_price_x96: {sqrt_price_x96}, tick: {tick})")
            
            return price
            
        except ContractLogicError as e:
            logger.error(f"Contract logic error getting price: {e}")
            return None
        except Web3Exception as e:
            logger.error(f"Web3 error getting price: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting price: {e}")
            return None
    
    def _sqrt_price_x96_to_price(self, sqrt_price_x96: int) -> float:
        """
        Convert sqrtPriceX96 to actual price
        
        Args:
            sqrt_price_x96: Square root of price in X96 format
            
        Returns:
            Actual price as float
        """
        # sqrtPriceX96 = sqrt(price) * 2^96
        # price = (sqrtPriceX96 / 2^96)^2
        
        Q96 = 2**96
        sqrt_price = sqrt_price_x96 / Q96
        price = sqrt_price ** 2
        
        # Adjust for token decimals (USDC has 6 decimals, WETH has 18)
        # Price needs to be adjusted by 10^(decimal1 - decimal0)
        if self.is_eth_token0:
            # ETH (18 decimals) / USDC (6 decimals)
            price = price * (10 ** (6 - 18))
        else:
            # USDC (6 decimals) / ETH (18 decimals)  
            price = price * (10 ** (18 - 6))
        
        return price
    
    def price_to_tick(self, price: float) -> int:
        """
        Convert price to tick
        
        Args:
            price: Price to convert
            
        Returns:
            Corresponding tick
        """
        # If ETH is token1, invert price for tick calculation
        if not self.is_eth_token0:
            price = 1 / price
        
        # Adjust for token decimals
        if self.is_eth_token0:
            adjusted_price = price / (10 ** (6 - 18))
        else:
            adjusted_price = price / (10 ** (18 - 6))
        
        # tick = log_1.0001(price)
        tick = math.log(adjusted_price) / math.log(1.0001)
        
        return int(round(tick))
    
    def tick_to_price(self, tick: int) -> float:
        """
        Convert tick to price
        
        Args:
            tick: Tick to convert
            
        Returns:
            Corresponding price
        """
        # price = 1.0001^tick
        price = (1.0001 ** tick)
        
        # Adjust for token decimals
        if self.is_eth_token0:
            price = price * (10 ** (6 - 18))
        else:
            price = price * (10 ** (18 - 6))
        
        # If ETH is token1, invert price
        if not self.is_eth_token0:
            price = 1 / price
            
        return price
    
    def get_nearest_usable_tick(self, tick: int, tick_spacing: int = 60) -> int:
        """
        Get nearest usable tick (must be divisible by tick spacing)
        
        Args:
            tick: Raw tick
            tick_spacing: Tick spacing for the fee tier (60 for 0.3% fee)
            
        Returns:
            Nearest usable tick
        """
        return round(tick / tick_spacing) * tick_spacing
    
    def get_tick_range_for_price_range(self, lower_price: float, upper_price: float) -> Tuple[int, int]:
        """
        Get tick range for given price range
        
        Args:
            lower_price: Lower bound price
            upper_price: Upper bound price
            
        Returns:
            Tuple of (lower_tick, upper_tick)
        """
        lower_tick = self.price_to_tick(lower_price)
        upper_tick = self.price_to_tick(upper_price)
        
        # Ensure proper ordering
        if lower_tick > upper_tick:
            lower_tick, upper_tick = upper_tick, lower_tick
        
        # Make ticks usable (divisible by tick spacing)
        tick_spacing = 60  # For 0.3% fee tier
        lower_tick = self.get_nearest_usable_tick(lower_tick, tick_spacing)
        upper_tick = self.get_nearest_usable_tick(upper_tick, tick_spacing)
        
        # Ensure we have a valid range
        if lower_tick >= upper_tick:
            upper_tick = lower_tick + tick_spacing
        
        logger.debug(f"Price range [{lower_price:.6f}, {upper_price:.6f}] -> Tick range [{lower_tick}, {upper_tick}]")
        
        return lower_tick, upper_tick
    
    def get_pool_info(self) -> dict:
        """Get pool information"""
        try:
            slot0_result = self.pool_contract.functions.slot0().call()
            current_price = self.get_current_price()
            
            return {
                "pool_address": self.pool_address,
                "token0": self.token0,
                "token1": self.token1,
                "fee": self.fee,
                "is_eth_token0": self.is_eth_token0,
                "sqrt_price_x96": slot0_result[0],
                "tick": slot0_result[1],
                "current_price": current_price,
                "observation_index": slot0_result[2],
                "observation_cardinality": slot0_result[3]
            }
        except Exception as e:
            logger.error(f"Error getting pool info: {e}")
            return {"error": str(e)} 