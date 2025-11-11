"""Live trading integration with Alpaca broker.

This module provides a live/paper trading interface for executing real (or paper)
trades via the Alpaca API. Includes risk management controls to prevent large losses.

WARNING: Paper trading only. Never use with real money without proper testing.
"""

import logging
import os
from typing import Optional, Dict
from datetime import datetime, time

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, OrderSide, TimeInForce
    from alpaca.trading.enums import OrderStatus
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed. Live trading features unavailable.")


class RiskManager:
    """Manages trading risk and validates orders.
    
    Attributes:
        trade_risk_percent (float): Max % of portfolio per trade (default: 1%).
        daily_loss_limit (float): Max daily loss % before halting (default: -2%).
        max_position_size (float): Max % per position (default: 5%).
    """
    
    def __init__(
        self,
        trade_risk_percent: float = 0.01,
        daily_loss_limit: float = -0.02,
        max_position_size: float = 0.05,
    ):
        """Initialize risk manager.
        
        Args:
            trade_risk_percent (float): Max % of portfolio per trade.
            daily_loss_limit (float): Max daily loss (negative) before halt.
            max_position_size (float): Max % per single position.
        """
        self.trade_risk_percent = trade_risk_percent
        self.daily_loss_limit = daily_loss_limit
        self.max_position_size = max_position_size
        self.daily_start_value = None
        self.daily_trades_count = 0
    
    def check_position_limit(self, portfolio_value: float, current_price: float) -> int:
        """Calculate max shares allowed based on risk percentage.
        
        Args:
            portfolio_value (float): Current total portfolio value.
            current_price (float): Price of asset.
            
        Returns:
            int: Maximum shares to trade.
        """
        risk_amount = portfolio_value * self.trade_risk_percent
        max_shares = int(risk_amount / current_price)
        
        # Also respect max position size
        max_position_shares = int(portfolio_value * self.max_position_size / current_price)
        
        return min(max_shares, max_position_shares)
    
    def check_daily_loss(self, current_value: float, starting_value: float) -> bool:
        """Check if daily loss limit exceeded.
        
        Args:
            current_value (float): Current portfolio value.
            starting_value (float): Portfolio value at market open.
            
        Returns:
            bool: True if trading should continue, False if halted.
        """
        daily_return = (current_value - starting_value) / starting_value
        
        if daily_return < self.daily_loss_limit:
            logger.warning(f"Daily loss limit hit: {daily_return:.2%}")
            return False
        
        return True
    
    def validate_trading_hours(self) -> bool:
        """Check if market is currently open (9:30 AM - 4:00 PM ET).
        
        Returns:
            bool: True if market is open, False otherwise.
        """
        now = datetime.now()
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        # Simple check (no holiday calendar)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        if market_open <= now.time() <= market_close:
            return True
        
        return False
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        current_price: float,
        portfolio_value: float,
    ) -> tuple:
        """Validate order before submission.
        
        Args:
            symbol (str): Stock ticker.
            side (str): 'buy' or 'sell'.
            qty (int): Number of shares.
            current_price (float): Current price.
            portfolio_value (float): Current portfolio value.
            
        Returns:
            tuple: (is_valid, message)
        """
        # Check market hours
        if not self.validate_trading_hours():
            return False, "Market is closed"
        
        # Check quantity
        if qty <= 0:
            return False, "Quantity must be positive"
        
        # Check position limit for buy orders
        if side.lower() == "buy":
            max_qty = self.check_position_limit(portfolio_value, current_price)
            if qty > max_qty:
                return False, f"Quantity {qty} exceeds limit {max_qty}"
        
        return True, "Valid"


class AlpacaTrader:
    """Live/paper trading client for Alpaca.
    
    Executes buy/sell orders via Alpaca API with risk management.
    
    Attributes:
        client (TradingClient): Alpaca trading client.
        risk_manager (RiskManager): Risk management instance.
        paper (bool): True for paper trading, False for live.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        risk_manager: Optional[RiskManager] = None,
    ):
        """Initialize Alpaca trader.
        
        Args:
            api_key (Optional[str]): Alpaca API key. If None, reads ALPACA_API_KEY env var.
            secret_key (Optional[str]): Alpaca secret key. If None, reads ALPACA_SECRET_KEY env var.
            paper (bool): True for paper trading (default: True). NEVER set to False without testing!
            risk_manager (Optional[RiskManager]): Risk manager instance. If None, creates default.
            
        Raises:
            RuntimeError: If Alpaca SDK not installed or credentials missing.
            ValueError: If paper=False (safety check for live trading).
        """
        if not ALPACA_AVAILABLE:
            raise RuntimeError(
                "alpaca-py not installed. Install with: pip install alpaca-py"
            )
        
        if not paper:
            raise ValueError(
                "Live trading (paper=False) is dangerous and disabled by default. "
                "Only enable after thorough testing with paper trading first."
            )
        
        api_key = api_key or os.getenv("ALPACA_API_KEY")
        secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError(
                "API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass as arguments."
            )
        
        self.paper = paper
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        
        self.client = TradingClient(api_key, secret_key, base_url=base_url)
        self.risk_manager = risk_manager or RiskManager()
        
        logger.info(f"Alpaca trader initialized ({'PAPER' if paper else 'LIVE'} mode)")
    
    def get_account_info(self) -> Dict:
        """Get current account information.
        
        Returns:
            Dict: Account info {
                'cash': float,
                'portfolio_value': float,
                'buying_power': float,
            }
        """
        try:
            account = self.client.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            raise
    
    def submit_buy_order(
        self,
        symbol: str,
        qty: int,
        current_price: float,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Submit a buy market order.
        
        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            qty (int): Number of shares.
            current_price (float): Current price (for validation).
            max_retries (int): Max retry attempts (default: 3).
            
        Returns:
            Optional[str]: Order ID if successful, None if failed.
        """
        try:
            account = self.get_account_info()
            
            # Validate order
            is_valid, msg = self.risk_manager.validate_order(
                symbol, "buy", qty, current_price, account['portfolio_value']
            )
            
            if not is_valid:
                logger.warning(f"Order validation failed: {msg}")
                return None
            
            # Submit order
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            
            order = self.client.submit_order(order_request)
            logger.info(f"BUY order placed: {symbol} x{qty} (ID: {order.id})")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Error submitting buy order: {e}")
            return None
    
    def submit_sell_order(
        self,
        symbol: str,
        qty: int,
        current_price: float,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Submit a sell market order.
        
        Args:
            symbol (str): Stock ticker.
            qty (int): Number of shares.
            current_price (float): Current price (for validation).
            max_retries (int): Max retry attempts (default: 3).
            
        Returns:
            Optional[str]: Order ID if successful, None if failed.
        """
        try:
            account = self.get_account_info()
            
            # Validate order
            is_valid, msg = self.risk_manager.validate_order(
                symbol, "sell", qty, current_price, account['portfolio_value']
            )
            
            if not is_valid:
                logger.warning(f"Order validation failed: {msg}")
                return None
            
            # Submit order
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            
            order = self.client.submit_order(order_request)
            logger.info(f"SELL order placed: {symbol} x{qty} (ID: {order.id})")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Error submitting sell order: {e}")
            return None
    
    def get_positions(self) -> Dict[str, float]:
        """Get all current positions.
        
        Returns:
            Dict[str, float]: Dictionary of {symbol: quantity}.
        """
        try:
            positions = self.client.get_all_positions()
            return {pos.symbol: float(pos.qty) for pos in positions}
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}
    
    def close_position(self, symbol: str) -> Optional[str]:
        """Close all shares of a position.
        
        Args:
            symbol (str): Stock ticker.
            
        Returns:
            Optional[str]: Order ID if successful, None if failed.
        """
        try:
            positions = self.get_positions()
            if symbol not in positions:
                logger.warning(f"No position in {symbol}")
                return None
            
            qty = int(positions[symbol])
            if qty == 0:
                return None
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            
            order = self.client.submit_order(order_request)
            logger.info(f"Position closed: {symbol} x{qty} (ID: {order.id})")
            
            return order.id
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
