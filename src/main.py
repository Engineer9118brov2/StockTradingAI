"""Command-line interface for StockTradingAI.

Main entry point for the trading bot with commands for fetching data,
training models, and running backtests.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_fetcher import DataFetcher
from src.backtester import Backtester, SimpleBacktester, print_metrics_table
from src.strategies.rule_based import SMACrossoverStrategy, SimpleRuleBasedBacktest
from src.strategies.ml_based import MLPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_command(args) -> pd.DataFrame:
    """Fetch stock data from Polygon API.
    
    Args:
        args: Parsed arguments with ticker, days, and save options.
        
    Returns:
        pd.DataFrame: Fetched and preprocessed data.
    """
    logger.info(f"Fetching {args.days} days of data for {args.ticker}...")
    
    try:
        fetcher = DataFetcher()
        df = fetcher.fetch_data(ticker=args.ticker, days=args.days)
        
        # Preprocess
        df_processed = DataFetcher.preprocess_data(df)
        
        logger.info(f"Data shape: {df_processed.shape}")
        logger.info(f"Columns: {list(df_processed.columns)}")
        
        return df_processed
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        sys.exit(1)


def train_command(args) -> MLPredictor:
    """Train ML model on stock data.
    
    Args:
        args: Parsed arguments with data file path.
        
    Returns:
        MLPredictor: Trained ML predictor.
    """
    logger.info("Training ML model...")
    
    try:
        # Load data
        if not Path(args.data).exists():
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)
        
        df = pd.read_csv(args.data)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Train model
        predictor = MLPredictor()
        metrics = predictor.train_model(df)
        
        logger.info(f"Model trained successfully!")
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        
        return predictor
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        sys.exit(1)


def backtest_command(args):
    """Run backtest on stock data.
    
    Args:
        args: Parsed arguments with strategy, data file, and plot options.
    """
    logger.info(f"Running {args.strategy} backtest...")
    
    try:
        # Load data
        if not Path(args.data).exists():
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)
        
        df = pd.read_csv(args.data)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        if args.strategy.lower() == "rule":
            _backtest_rule_based(df, args)
        elif args.strategy.lower() == "ml":
            _backtest_ml_based(df, args)
        else:
            logger.error(f"Unknown strategy: {args.strategy}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        sys.exit(1)


def _backtest_rule_based(df: pd.DataFrame, args):
    """Run rule-based (SMA crossover) backtest.
    
    Args:
        df (pd.DataFrame): Stock data.
        args: Parsed arguments.
    """
    logger.info("Running SMA Crossover backtest...")
    
    # Use simple backtester (fallback if backtrader not available)
    backtester = SimpleRuleBasedBacktest(
        df,
        short_window=args.short_window,
        long_window=args.long_window,
        trade_size=args.trade_size,
        stop_loss_pct=args.stop_loss_pct,
        initial_cash=args.initial_cash,
    )
    
    result = backtester.run()
    portfolio_values = result["portfolio_value"]
    trades = result["trades"]
    
    # Compute metrics
    metrics = SimpleBacktester.compute_metrics(portfolio_values, args.initial_cash)
    metrics["trades"] = len([t for t in trades if t.get("type") == "buy"])
    
    # Print results
    logger.info("\n" + print_metrics_table(metrics))
    logger.info(f"Total trades: {metrics['trades']}")
    
    # Plot if requested
    if args.plot:
        plot_path = args.plot_path or f"examples/{args.ticker}_backtest_rule.png"
        SimpleBacktester.plot_equity_curve(
            portfolio_values,
            title=f"SMA Crossover Backtest - {args.ticker}",
            save_path=plot_path if args.plot_path or not Path(plot_path).parent.exists() else None,
        )


def _backtest_ml_based(df: pd.DataFrame, args):
    """Run ML-based backtest.
    
    Args:
        df (pd.DataFrame): Stock data.
        args: Parsed arguments.
    """
    logger.info("Running ML-based backtest...")
    
    # Load or train model
    try:
        predictor = MLPredictor()
        predictor.load_model()
        logger.info("Loaded existing model")
    except FileNotFoundError:
        logger.info("Model not found, training new model...")
        metrics = predictor.train_model(df)
    
    # Get predictions
    signals = predictor.predict(df, threshold=0.5)
    
    # Simple manual backtest using signals
    cash = args.initial_cash
    shares = 0
    portfolio_values = []
    
    for i, row in df.iterrows():
        current_price = row["close"]
        
        # Buy signal
        if signals[i] == 1 and shares == 0:
            shares = args.trade_size
            cash -= shares * current_price
            logger.debug(f"Buy at {current_price:.2f}")
        
        # Sell signal
        elif signals[i] == 0 and shares > 0:
            cash += shares * current_price
            logger.debug(f"Sell at {current_price:.2f}")
            shares = 0
        
        portfolio_values.append(cash + shares * current_price)
    
    # Compute metrics
    metrics = SimpleBacktester.compute_metrics(portfolio_values, args.initial_cash)
    
    # Print results
    logger.info("\n" + print_metrics_table(metrics))
    
    # Plot if requested
    if args.plot:
        plot_path = f"examples/{args.ticker}_backtest_ml.png"
        SimpleBacktester.plot_equity_curve(
            portfolio_values,
            title=f"ML-based Backtest - {args.ticker}",
            save_path=plot_path,
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="StockTradingAI: An educational algorithmic trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data
  python main.py fetch --ticker AAPL --days 365

  # Train ML model
  python main.py train --data data/AAPL_data.csv

  # Run backtest
  python main.py backtest --strategy rule --data data/AAPL_data.csv --plot

  # Run ML backtest
  python main.py backtest --strategy ml --data data/AAPL_data.csv --plot
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch stock data from Polygon API")
    fetch_parser.add_argument("--ticker", default="AAPL", help="Stock ticker (default: AAPL)")
    fetch_parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--data", required=True, help="Path to data CSV file")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "--strategy",
        choices=["rule", "ml"],
        default="rule",
        help="Strategy to backtest (default: rule)",
    )
    backtest_parser.add_argument("--data", required=True, help="Path to data CSV file")
    backtest_parser.add_argument("--ticker", default="AAPL", help="Stock ticker (default: AAPL)")
    backtest_parser.add_argument(
        "--initial-cash",
        type=float,
        default=10000,
        help="Initial cash amount (default: 10000)",
    )
    backtest_parser.add_argument(
        "--short-window",
        type=int,
        default=50,
        help="Short SMA window (default: 50)",
    )
    backtest_parser.add_argument(
        "--long-window",
        type=int,
        default=200,
        help="Long SMA window (default: 200)",
    )
    backtest_parser.add_argument(
        "--trade-size",
        type=int,
        default=10,
        help="Shares per trade (default: 10)",
    )
    backtest_parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.05,
        help="Stop-loss percentage (default: 0.05 = 5%%)",
    )
    backtest_parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot results",
    )
    backtest_parser.add_argument(
        "--plot-path",
        help="Path to save plot",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "fetch":
        fetch_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "backtest":
        backtest_command(args)


if __name__ == "__main__":
    main()
