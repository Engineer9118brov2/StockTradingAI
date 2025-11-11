"""Backtesting engine for trading strategies.

This module provides tools to run backtests, calculate performance metrics,
and visualize results.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import backtrader as bt
    import matplotlib.pyplot as plt
except ImportError:
    bt = None
    plt = None

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine using backtrader.
    
    Runs backtests for trading strategies and computes performance metrics.
    """
    
    def __init__(self, initial_cash: float = 10000):
        """Initialize backtester.
        
        Args:
            initial_cash (float): Initial portfolio cash (default: $10,000).
        """
        self.initial_cash = initial_cash
        self.results = None
        self.cerebro = None
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy_class,
        strategy_params: Optional[Dict] = None,
        plot: bool = False,
        plot_path: Optional[str] = None,
    ) -> Dict:
        """Run backtest using backtrader.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data. Must have columns:
                [date, open, high, low, close, volume]
            strategy_class: Strategy class (e.g., SMACrossoverStrategy).
            strategy_params (Optional[Dict]): Parameters to pass to strategy.
            plot (bool): Whether to plot results (requires matplotlib).
            plot_path (Optional[str]): Path to save plot. If None, displays plot.
            
        Returns:
            Dict: Performance metrics {
                'total_return': float (in %),
                'sharpe_ratio': float,
                'max_drawdown': float (in %),
                'final_value': float,
                'trades': int,
                'start_value': float,
            }
            
        Raises:
            RuntimeError: If backtrader is not installed.
        """
        if bt is None:
            logger.error("backtrader not installed. Install with: pip install backtrader")
            raise RuntimeError("backtrader not installed")
        
        logger.info("Starting backtest...")
        
        # Prepare data
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        
        # Create Cerebro instance
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Create data feed
        data = bt.feeds.PandasData(dataname=df)
        self.cerebro.adddata(data)
        
        # Add strategy
        if strategy_params is None:
            strategy_params = {}
        self.cerebro.addstrategy(strategy_class, **strategy_params)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        
        # Run backtest
        try:
            results = self.cerebro.run()
            strat = results[0]
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise RuntimeError(f"Backtest failed: {str(e)}")
        
        # Extract metrics
        metrics = self._extract_metrics(strat, self.cerebro.broker.getvalue())
        
        # Plot if requested
        if plot:
            self._plot_results(plot_path)
        
        logger.info(f"Backtest complete. Total return: {metrics['total_return']:.2f}%")
        
        return metrics
    
    def _extract_metrics(self, strat, final_value: float) -> Dict:
        """Extract performance metrics from strategy results.
        
        Args:
            strat: Strategy instance with analyzers.
            final_value (float): Final portfolio value.
            
        Returns:
            Dict: Performance metrics.
        """
        metrics = {
            "start_value": self.initial_cash,
            "final_value": final_value,
            "total_return": ((final_value - self.initial_cash) / self.initial_cash) * 100,
            "trades": 0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        
        # Get analyzer results
        try:
            returns_analyzer = strat.analyzers.returns
            if returns_analyzer is not None and hasattr(returns_analyzer, "rtot"):
                metrics["total_return"] = returns_analyzer.rtot * 100
        except:
            pass
        
        try:
            sharpe_analyzer = strat.analyzers.sharpe
            if sharpe_analyzer is not None and hasattr(sharpe_analyzer, "sharperatio"):
                metrics["sharpe_ratio"] = sharpe_analyzer.sharperatio or 0.0
        except:
            pass
        
        try:
            drawdown_analyzer = strat.analyzers.drawdown
            if drawdown_analyzer is not None and hasattr(drawdown_analyzer, "max"):
                metrics["max_drawdown"] = drawdown_analyzer.max.drawdown if drawdown_analyzer.max else 0.0
        except:
            pass
        
        # Count trades
        if hasattr(strat, "trades"):
            metrics["trades"] = len([t for t in strat.trades if t.get("type") == "buy"])
        
        return metrics
    
    def _plot_results(self, plot_path: Optional[str] = None):
        """Plot backtest results.
        
        Args:
            plot_path (Optional[str]): Path to save plot. If None, displays plot.
        """
        if plt is None or self.cerebro is None:
            logger.warning("matplotlib not available for plotting")
            return
        
        try:
            self.cerebro.plot()
            
            if plot_path:
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                logger.info(f"Plot saved to {plot_path}")
            else:
                plt.show()
        except Exception as e:
            logger.warning(f"Could not plot results: {str(e)}")


class SimpleBacktester:
    """Simple backtester without backtrader (for fallback/comparison).
    
    Computes performance metrics on a list of portfolio values.
    """
    
    @staticmethod
    def compute_metrics(
        portfolio_values: List[float],
        initial_cash: float = 10000,
        risk_free_rate: float = 0.02,
    ) -> Dict:
        """Compute performance metrics from portfolio values.
        
        Args:
            portfolio_values (List[float]): Daily portfolio values.
            initial_cash (float): Initial cash amount.
            risk_free_rate (float): Annual risk-free rate (default: 2%).
            
        Returns:
            Dict: Performance metrics {
                'total_return': float (in %),
                'sharpe_ratio': float,
                'max_drawdown': float (in %),
                'final_value': float,
            }
        """
        pv = np.array(portfolio_values)
        
        if len(pv) == 0:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "final_value": initial_cash,
            }
        
        # Total return
        final_value = pv[-1]
        total_return = ((final_value - initial_cash) / initial_cash) * 100
        
        # Daily returns
        returns = np.diff(pv) / pv[:-1]
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and np.std(returns) > 0:
            excess_return = np.mean(returns) - (risk_free_rate / 252)
            sharpe = (excess_return / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": final_value,
        }
    
    @staticmethod
    def plot_equity_curve(
        portfolio_values: List[float],
        dates: Optional[List] = None,
        title: str = "Portfolio Equity Curve",
        save_path: Optional[str] = None,
    ):
        """Plot equity curve.
        
        Args:
            portfolio_values (List[float]): Daily portfolio values.
            dates (Optional[List]): Dates corresponding to portfolio values.
            title (str): Plot title.
            save_path (Optional[str]): Path to save plot.
        """
        if plt is None:
            logger.warning("matplotlib not available for plotting")
            return
        
        plt.figure(figsize=(12, 6))
        
        if dates is not None:
            plt.plot(dates, portfolio_values, linewidth=2, label="Portfolio Value")
        else:
            plt.plot(portfolio_values, linewidth=2, label="Portfolio Value")
        
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Equity curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def print_metrics_table(metrics: Dict) -> str:
    """Format performance metrics as a table string.
    
    Args:
        metrics (Dict): Performance metrics dictionary.
        
    Returns:
        str: Formatted table string.
    """
    table = []
    table.append("+" + "-" * 40 + "+")
    table.append("| Metric                   | Value      |")
    table.append("+" + "-" * 40 + "+")
    
    for key, value in metrics.items():
        if key == "final_value" or key == "start_value":
            formatted = f"${value:,.2f}"
        elif key == "total_return" or key == "max_drawdown":
            formatted = f"{value:.2f}%"
        elif key == "sharpe_ratio":
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        
        key_display = key.replace("_", " ").title()
        table.append(f"| {key_display:<24} | {formatted:>9} |")
    
    table.append("+" + "-" * 40 + "+")
    
    return "\n".join(table)
