"""Rule-based SMA crossover trading strategy.

This module implements a simple moving average crossover strategy:
- Buy when short SMA (50-day) crosses above long SMA (200-day)
- Sell when short SMA crosses below long SMA
- Implements stop-loss at 5% below buy price
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

try:
    import backtrader as bt
except ImportError:
    bt = None

logger = logging.getLogger(__name__)


class SMACrossoverStrategy(bt.Strategy if bt else object):
    """SMA Crossover strategy using backtrader.
    
    A strategy that buys when short SMA crosses above long SMA and sells on reverse crossover.
    Includes stop-loss protection.
    
    Params:
        short_window (int): Short SMA period (default: 50)
        long_window (int): Long SMA period (default: 200)
        trade_size (int): Number of shares per trade (default: 10)
        stop_loss_pct (float): Stop-loss percentage (default: 0.05 = 5%)
    """
    
    params = {
        "short_window": 50,
        "long_window": 200,
        "trade_size": 10,
        "stop_loss_pct": 0.05,
    }
    
    def __init__(self):
        """Initialize the strategy."""
        if bt is None:
            raise RuntimeError("backtrader not installed. Install it with: pip install backtrader")
        
        # Calculate SMAs
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_window
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_window
        )
        
        # Track trades
        self.trades = []
        self.buy_price = None
        self.buy_date = None
        self.position_size = 0
        
        # Crossover indicator
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
    
    def next(self):
        """Execute trading logic for each bar.
        
        - Buy signal: short SMA > long SMA and not already in position
        - Sell signal: short SMA < long SMA or stop-loss hit
        """
        # Check stop-loss
        if self.position_size > 0 and self.buy_price is not None:
            stop_loss_price = self.buy_price * (1 - self.params.stop_loss_pct)
            if self.data.close[0] <= stop_loss_price:
                logger.info(
                    f"Stop-loss triggered at {self.data.close[0]:.2f} "
                    f"(bought at {self.buy_price:.2f})"
                )
                self.sell(size=self.position_size)
                self.trades.append({
                    "date": self.data.datetime.date(0),
                    "type": "sell",
                    "price": self.data.close[0],
                    "reason": "stop_loss",
                })
                self.position_size = 0
                self.buy_price = None
                return
        
        # Buy signal: crossover positive (short > long)
        if self.crossover[0] > 0 and self.position_size == 0:
            self.buy(size=self.params.trade_size)
            self.buy_price = self.data.close[0]
            self.buy_date = self.data.datetime.date(0)
            self.position_size = self.params.trade_size
            self.trades.append({
                "date": self.buy_date,
                "type": "buy",
                "price": self.buy_price,
            })
            logger.info(
                f"Buy signal at {self.buy_date}: {self.params.trade_size} shares @ "
                f"${self.buy_price:.2f}"
            )
        
        # Sell signal: crossover negative (short < long)
        elif self.crossover[0] < 0 and self.position_size > 0:
            self.sell(size=self.position_size)
            self.trades.append({
                "date": self.data.datetime.date(0),
                "type": "sell",
                "price": self.data.close[0],
                "reason": "crossover",
            })
            logger.info(
                f"Sell signal at {self.data.datetime.date(0)}: "
                f"{self.position_size} shares @ ${self.data.close[0]:.2f}"
            )
            self.position_size = 0
            self.buy_price = None


class SimpleRuleBasedBacktest:
    """Simple rule-based backtesting without backtrader (fallback).
    
    Implements a manual SMA crossover backtest for comparison.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        short_window: int = 50,
        long_window: int = 200,
        trade_size: int = 10,
        stop_loss_pct: float = 0.05,
        initial_cash: float = 10000,
    ):
        """Initialize the simple backtester.
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' column.
            short_window (int): Short SMA period.
            long_window (int): Long SMA period.
            trade_size (int): Shares per trade.
            stop_loss_pct (float): Stop-loss percentage.
            initial_cash (float): Initial portfolio cash.
        """
        self.df = df.copy()
        self.short_window = short_window
        self.long_window = long_window
        self.trade_size = trade_size
        self.stop_loss_pct = stop_loss_pct
        self.initial_cash = initial_cash
        
        self.trades = []
    
    def run(self) -> Dict:
        """Run the backtest.
        
        Returns:
            Dict: Dictionary with 'portfolio_value' (list) and 'trades' (list).
        """
        # Calculate SMAs
        self.df["sma_short"] = self.df["close"].rolling(self.short_window).mean()
        self.df["sma_long"] = self.df["close"].rolling(self.long_window).mean()
        
        # Initialize portfolio
        cash = self.initial_cash
        shares = 0
        buy_price = None
        portfolio_values = []
        
        for idx, row in self.df.iterrows():
            current_price = row["close"]
            
            # Check stop-loss
            if shares > 0 and buy_price is not None:
                stop_loss_price = buy_price * (1 - self.stop_loss_pct)
                if current_price <= stop_loss_price:
                    cash += shares * current_price
                    self.trades.append({
                        "date": idx,
                        "type": "sell",
                        "price": current_price,
                        "reason": "stop_loss",
                    })
                    shares = 0
                    buy_price = None
            
            # Check for NaN SMAs
            if pd.isna(row["sma_short"]) or pd.isna(row["sma_long"]):
                portfolio_values.append(cash + shares * current_price)
                continue
            
            # Buy signal
            if row["sma_short"] > row["sma_long"] and shares == 0:
                shares = self.trade_size
                cash -= shares * current_price
                buy_price = current_price
                self.trades.append({
                    "date": idx,
                    "type": "buy",
                    "price": current_price,
                })
            
            # Sell signal
            elif row["sma_short"] < row["sma_long"] and shares > 0:
                cash += shares * current_price
                self.trades.append({
                    "date": idx,
                    "type": "sell",
                    "price": current_price,
                    "reason": "crossover",
                })
                shares = 0
                buy_price = None
            
            # Update portfolio value
            portfolio_values.append(cash + shares * current_price)
        
        return {
            "portfolio_value": portfolio_values,
            "trades": self.trades,
        }
