"""
Position Manager for Uniswap V3 NFT positions
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3Exception
from eth_account import Account

from config.contracts import (
    POSITION_MANAGER_ABI, NONFUNGIBLE_POSITION_MANAGER,
    ERC20_ABI, WETH9, USDC
)
from config.settings import settings

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, web3: Web3):
        """
        Initialize Position Manager
        
        Args:
            web3: Web3 instance
        """
        self.web3 = web3
        self.account = Account.from_key(settings.PRIVATE_KEY)
        self.position_manager = web3.eth.contract(
            address=NONFUNGIBLE_POSITION_MANAGER,
            abi=POSITION_MANAGER_ABI
        )
        
        # Token contracts
        self.weth_contract = web3.eth.contract(address=WETH9, abi=ERC20_ABI)
        self.usdc_contract = web3.eth.contract(address=USDC, abi=ERC20_ABI)
        
        self.current_position_id: Optional[int] = None
        
    def get_token_balances(self) -> Tuple[float, float]:
        """
        Get current token balances
        
        Returns:
            Tuple of (eth_balance, usdc_balance)
        """
        try:
            # Get ETH balance
            eth_balance_wei = self.web3.eth.get_balance(settings.PUBLIC_ADDRESS)
            eth_balance = self.web3.from_wei(eth_balance_wei, 'ether')
            
            # Get USDC balance
            usdc_balance_raw = self.usdc_contract.functions.balanceOf(settings.PUBLIC_ADDRESS).call()
            usdc_balance = usdc_balance_raw / (10 ** 6)  # USDC has 6 decimals
            
            logger.debug(f"Balances - ETH: {eth_balance:.6f}, USDC: {usdc_balance:.2f}")
            
            return float(eth_balance), float(usdc_balance)
            
        except Exception as e:
            logger.error(f"Error getting token balances: {e}")
            return 0.0, 0.0
    
    def approve_tokens(self, eth_amount: float, usdc_amount: float) -> bool:
        """
        Approve tokens for position manager
        
        Args:
            eth_amount: Amount of ETH to approve
            usdc_amount: Amount of USDC to approve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert amounts to wei/raw units
            eth_amount_wei = self.web3.to_wei(eth_amount, 'ether')
            usdc_amount_raw = int(usdc_amount * (10 ** 6))
            
            transactions = []
            
            # Check WETH allowance
            weth_allowance = self.weth_contract.functions.allowance(
                settings.PUBLIC_ADDRESS, NONFUNGIBLE_POSITION_MANAGER
            ).call()
            
            if weth_allowance < eth_amount_wei:
                weth_approve_tx = self.weth_contract.functions.approve(
                    NONFUNGIBLE_POSITION_MANAGER,
                    eth_amount_wei
                ).build_transaction({
                    'from': settings.PUBLIC_ADDRESS,
                    'gas': 100000,
                    'gasPrice': self.web3.to_wei(20, 'gwei'),
                    'nonce': self.web3.eth.get_transaction_count(settings.PUBLIC_ADDRESS)
                })
                transactions.append(("WETH Approve", weth_approve_tx))
            
            # Check USDC allowance
            usdc_allowance = self.usdc_contract.functions.allowance(
                settings.PUBLIC_ADDRESS, NONFUNGIBLE_POSITION_MANAGER
            ).call()
            
            if usdc_allowance < usdc_amount_raw:
                usdc_approve_tx = self.usdc_contract.functions.approve(
                    NONFUNGIBLE_POSITION_MANAGER,
                    usdc_amount_raw
                ).build_transaction({
                    'from': settings.PUBLIC_ADDRESS,
                    'gas': 100000,
                    'gasPrice': self.web3.to_wei(20, 'gwei'),
                    'nonce': self.web3.eth.get_transaction_count(settings.PUBLIC_ADDRESS) + len(transactions)
                })
                transactions.append(("USDC Approve", usdc_approve_tx))
            
            # Send approval transactions
            for name, tx in transactions:
                signed_tx = self.web3.eth.account.sign_transaction(tx, settings.PRIVATE_KEY)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                logger.info(f"Sent {name} transaction: {tx_hash.hex()}")
                
                # Wait for confirmation
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
                if receipt.status == 0:
                    logger.error(f"{name} transaction failed")
                    return False
                    
                logger.info(f"{name} confirmed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error approving tokens: {e}")
            return False

    def mint_position(
        self,
        tick_lower: int,
        tick_upper: int,
        eth_amount: float,
        usdc_amount: float,
        fee: int = 3000
    ) -> Optional[int]:
        """
        Mint new liquidity position
        
        Args:
            tick_lower: Lower tick of the position
            tick_upper: Upper tick of the position
            eth_amount: Amount of ETH to provide
            usdc_amount: Amount of USDC to provide
            fee: Pool fee tier (3000 = 0.3%)
            
        Returns:
            Position token ID or None if failed
        """
        try:
            # Convert amounts to wei/raw units
            eth_amount_wei = self.web3.to_wei(eth_amount, 'ether')
            usdc_amount_raw = int(usdc_amount * (10 ** 6))
            
            # Calculate minimum amounts (allow 5% slippage)
            eth_min = int(eth_amount_wei * 0.95)
            usdc_min = int(usdc_amount_raw * 0.95)
            
            # Build mint parameters
            mint_params = {
                'token0': WETH9,
                'token1': USDC,
                'fee': fee,
                'tickLower': tick_lower,
                'tickUpper': tick_upper,
                'amount0Desired': eth_amount_wei,
                'amount1Desired': usdc_amount_raw,
                'amount0Min': eth_min,
                'amount1Min': usdc_min,
                'recipient': settings.PUBLIC_ADDRESS,
                'deadline': int(time.time()) + 3600  # 1 hour from now
            }
            
            # Build transaction
            mint_tx = self.position_manager.functions.mint(mint_params).build_transaction({
                'from': settings.PUBLIC_ADDRESS,
                'gas': 500000,
                'gasPrice': self.web3.to_wei(30, 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(settings.PUBLIC_ADDRESS)
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(mint_tx, settings.PRIVATE_KEY)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Sent mint transaction: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt.status == 0:
                logger.error("Mint transaction failed")
                return None
            
            # Extract position ID from logs
            position_id = None
            for log in receipt.logs:
                try:
                    # Look for Transfer event (NFT mint)
                    if log.address.lower() == NONFUNGIBLE_POSITION_MANAGER.lower():
                        if len(log.topics) >= 4:
                            position_id = int(log.topics[3].hex(), 16)
                            break
                except:
                    continue
            
            if position_id:
                self.current_position_id = position_id
                logger.info(f"Position minted successfully. Token ID: {position_id}")
                logger.info(f"Gas used: {receipt.gasUsed}")
                return position_id
            else:
                logger.error("Could not extract position ID from transaction receipt")
                return None
                
        except Exception as e:
            logger.error(f"Error minting position: {e}")
            return None

    def remove_liquidity(self, position_id: int, liquidity_percent: float = 100.0) -> bool:
        """
        Remove liquidity from position
        
        Args:
            position_id: Position token ID
            liquidity_percent: Percentage of liquidity to remove (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get position info
            position_info = self.position_manager.functions.positions(position_id).call()
            current_liquidity = position_info[7]  # liquidity is at index 7
            
            if current_liquidity == 0:
                logger.warning(f"Position {position_id} has no liquidity")
                return True
            
            # Calculate liquidity to remove
            liquidity_to_remove = int(current_liquidity * (liquidity_percent / 100.0))
            
            # Build decrease liquidity parameters
            decrease_params = {
                'tokenId': position_id,
                'liquidity': liquidity_to_remove,
                'amount0Min': 0,  # Accept any amount (could be improved with slippage protection)
                'amount1Min': 0,
                'deadline': int(time.time()) + 3600
            }
            
            # Build transaction
            decrease_tx = self.position_manager.functions.decreaseLiquidity(decrease_params).build_transaction({
                'from': settings.PUBLIC_ADDRESS,
                'gas': 300000,
                'gasPrice': self.web3.to_wei(30, 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(settings.PUBLIC_ADDRESS)
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(decrease_tx, settings.PRIVATE_KEY)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"Sent decrease liquidity transaction: {tx_hash.hex()}")
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if receipt.status == 0:
                logger.error("Decrease liquidity transaction failed")
                return False
            
            logger.info(f"Liquidity removed successfully. Gas used: {receipt.gasUsed}")
            
            # If we removed all liquidity, clear current position
            if liquidity_percent >= 100.0:
                self.current_position_id = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing liquidity: {e}")
            return False

    def get_position_info(self, position_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed position information"""
        try:
            position_data = self.position_manager.functions.positions(position_id).call()
            
            return {
                'position_id': position_id,
                'nonce': position_data[0],
                'operator': position_data[1],
                'token0': position_data[2],
                'token1': position_data[3],
                'fee': position_data[4],
                'tick_lower': position_data[5],
                'tick_upper': position_data[6],
                'liquidity': position_data[7],
                'fee_growth_inside0_last_x128': position_data[8],
                'fee_growth_inside1_last_x128': position_data[9],
                'tokens_owed0': position_data[10],
                'tokens_owed1': position_data[11]
            }
            
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None 