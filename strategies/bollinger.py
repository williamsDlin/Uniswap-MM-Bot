"""
Bollinger Bands strategy for Uniswap V3 liquidity management
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class BollingerBandsStrategy:
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands strategy
        
        Args:
            window: Number of periods for moving average calculation
            std_dev: Number of standard deviations for band calculation
        """
        self.window = window
        self.std_dev = std_dev
        self.price_history: List[float] = []
        
    def add_price(self, price: float) -> None:
        """Add a new price to the history"""
        self.price_history.append(price)
        
        # Keep only the required window size
        if len(self.price_history) > self.window * 2:
            self.price_history = self.price_history[-self.window * 2:]
    
    def calculate_bands(self) -> Optional[Tuple[float, float, float]]:
        """
        Calculate Bollinger Bands
        
        Returns:
            Tuple of (lower_band, middle_band, upper_band) or None if insufficient data
        """
        if len(self.price_history) < self.window:
            logger.warning(f"Insufficient price history: {len(self.price_history)} < {self.window}")
            return None
        
        # Use the last 'window' prices
        recent_prices = self.price_history[-self.window:]
        prices_series = pd.Series(recent_prices)
        
        # Calculate moving average and standard deviation
        middle_band = prices_series.mean()
        std = prices_series.std()
        
        # Calculate bands
        upper_band = middle_band + (self.std_dev * std)
        lower_band = middle_band - (self.std_dev * std)
        
        logger.debug(f"Bollinger Bands - Lower: {lower_band:.6f}, Middle: {middle_band:.6f}, Upper: {upper_band:.6f}")
        
        return lower_band, middle_band, upper_band
    
    def should_rebalance(self, current_price: float, current_range: Optional[Tuple[float, float]] = None) -> Tuple[bool, str]:
        """
        Determine if rebalancing is needed
        
        Args:
            current_price: Current market price
            current_range: Current position range (lower_price, upper_price)
            
        Returns:
            Tuple of (should_rebalance, reason)
        """
        bands = self.calculate_bands()
        if bands is None:
            return False, "Insufficient price history"
        
        lower_band, middle_band, upper_band = bands
        
        # If no current position, we should establish one
        if current_range is None:
            return True, "No active position"
        
        current_lower, current_upper = current_range
        
        # Check if current price is outside current range
        if current_price <= current_lower or current_price >= current_upper:
            return True, f"Price {current_price:.6f} outside current range [{current_lower:.6f}, {current_upper:.6f}]"
        
        # Check if current range is significantly different from Bollinger bands
        range_deviation_threshold = 0.1  # 10% deviation threshold
        
        lower_deviation = abs(current_lower - lower_band) / lower_band
        upper_deviation = abs(current_upper - upper_band) / upper_band
        
        if lower_deviation > range_deviation_threshold or upper_deviation > range_deviation_threshold:
            return True, f"Range deviation too high: lower {lower_deviation:.2%}, upper {upper_deviation:.2%}"
        
        return False, "Position within acceptable range"
    
    def get_optimal_range(self, current_price: float) -> Tuple[float, float]:
        """
        Get optimal price range for liquidity position
        
        Args:
            current_price: Current market price
            
        Returns:
            Tuple of (lower_price, upper_price)
        """
        bands = self.calculate_bands()
        if bands is None:
            # Fallback to a simple range around current price if no history
            range_width = current_price * 0.1  # 10% range
            return current_price - range_width, current_price + range_width
        
        lower_band, middle_band, upper_band = bands
        
        # Use Bollinger bands as the range, but ensure current price is within range
        lower_price = min(lower_band, current_price * 0.95)  # At least 5% below current
        upper_price = max(upper_band, current_price * 1.05)  # At least 5% above current
        
        logger.info(f"Optimal range: [{lower_price:.6f}, {upper_price:.6f}] (current: {current_price:.6f})")
        
        return lower_price, upper_price
    
    def get_position_size(self, available_eth: float, current_price: float) -> Tuple[float, float]:
        """
        Calculate optimal position size based on available funds
        
        Args:
            available_eth: Available ETH balance
            current_price: Current ETH/USDC price
            
        Returns:
            Tuple of (eth_amount, usdc_amount)
        """
        # Use a conservative portion of available funds
        max_eth_to_use = min(available_eth * 0.8, 0.01)  # Max 80% of balance or 0.01 ETH
        
        # For balanced liquidity, we'll use equal value in both tokens
        eth_amount = max_eth_to_use / 2
        usdc_amount = eth_amount * current_price
        
        logger.info(f"Position size: {eth_amount:.6f} ETH, {usdc_amount:.2f} USDC")
        
        return eth_amount, usdc_amount
    
    def get_statistics(self) -> dict:
        """Get strategy statistics"""
        if len(self.price_history) < 2:
            return {"status": "insufficient_data"}
        
        recent_prices = self.price_history[-min(len(self.price_history), self.window):]
        prices_series = pd.Series(recent_prices)
        
        bands = self.calculate_bands()
        
        stats = {
            "price_count": len(self.price_history),
            "current_price": self.price_history[-1] if self.price_history else None,
            "mean_price": prices_series.mean(),
            "std_dev": prices_series.std(),
            "min_price": prices_series.min(),
            "max_price": prices_series.max(),
        }
        
        if bands:
            lower_band, middle_band, upper_band = bands
            stats.update({
                "lower_band": lower_band,
                "middle_band": middle_band,
                "upper_band": upper_band,
                "band_width": upper_band - lower_band,
                "band_width_percent": ((upper_band - lower_band) / middle_band) * 100
            })
        
        return stats 