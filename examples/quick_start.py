#!/usr/bin/env python
"""
Quick start example for StockTradingAI.

This script demonstrates the basic workflow:
1. Fetch data from Polygon API
2. Backtest rule-based strategy
3. Train and backtest ML strategy
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import DataFetcher
from src.strategies.rule_based import SimpleRuleBasedBacktest
from src.strategies.ml_based import MLPredictor
from src.backtester import SimpleBacktester, print_metrics_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_with_sample_data():
    """Run example with synthetic data (no API key required)."""
    import pandas as pd
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("EXAMPLE: StockTradingAI with Synthetic Data")
    logger.info("=" * 60)
    
    # Create synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
    close_prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices + np.random.randn(500) * 0.2,
        "high": close_prices + abs(np.random.randn(500) * 0.3),
        "low": close_prices - abs(np.random.randn(500) * 0.3),
        "close": close_prices,
        "volume": np.random.randint(1000000, 10000000, 500),
    })
    
    logger.info(f"\nGenerated synthetic data with {len(df)} daily bars")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # ===== EXAMPLE 1: Rule-Based Backtest =====
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 1: SMA Crossover Strategy Backtest")
    logger.info("=" * 60)
    
    backtester = SimpleRuleBasedBacktest(
        df,
        short_window=50,
        long_window=200,
        trade_size=10,
        stop_loss_pct=0.05,
        initial_cash=10000,
    )
    
    result = backtester.run()
    portfolio_values = result["portfolio_value"]
    trades = result["trades"]
    
    # Calculate metrics
    metrics = SimpleBacktester.compute_metrics(portfolio_values, 10000)
    metrics["trades"] = len([t for t in trades if t.get("type") == "buy"])
    
    logger.info("\nSMA Crossover Strategy Results:")
    logger.info(print_metrics_table(metrics))
    
    logger.info(f"\nTrade History (first 10):")
    for i, trade in enumerate(trades[:10]):
        logger.info(f"  {i+1}. {trade['type'].upper()} @ ${trade['price']:.2f}")
    
    # ===== EXAMPLE 2: ML-Based Backtest =====
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 2: ML-Based Strategy")
    logger.info("=" * 60)
    
    # Preprocess data for ML
    df_ml = df.copy()
    df_ml["close_lag1"] = df_ml["close"].shift(1)
    df_ml["close_lag2"] = df_ml["close"].shift(2)
    # Add more lags for realistic training
    for lag in range(1, 15):
        for col in ["open", "high", "low", "close"]:
            df_ml[f"{col}_lag{lag}"] = df_ml[col].shift(lag)
    
    # Add label
    df_ml["label"] = (df_ml["close"].shift(-1) > df_ml["close"]).astype(int)
    df_ml = df_ml.dropna()
    
    logger.info(f"\nTraining ML model on {len(df_ml)} samples...")
    logger.info(f"Features: 56 (14 lags × 4 OHLC columns)")
    
    # Train model
    predictor = MLPredictor(model_path="/tmp/example_model.pkl")
    train_metrics = predictor.train_model(df_ml)
    
    logger.info(f"Training complete!")
    logger.info(f"  Test Accuracy: {train_metrics['accuracy']:.4f}")
    logger.info(f"  Test Size: {train_metrics['test_size']}")
    
    # Get predictions
    signals = predictor.predict(df_ml, threshold=0.5)
    
    # Simple backtest with ML signals
    logger.info(f"\nBacktesting ML strategy...")
    cash = 10000
    shares = 0
    portfolio_values_ml = []
    buy_count = 0
    
    for idx, (i, row) in enumerate(df_ml.iterrows()):
        if idx >= len(signals):
            break
        
        current_price = row["close"]
        
        # Buy signal
        if signals[idx] == 1 and shares == 0:
            shares = 10
            cash -= shares * current_price
            buy_count += 1
        # Sell signal
        elif signals[idx] == 0 and shares > 0:
            cash += shares * current_price
            shares = 0
        
        portfolio_values_ml.append(cash + shares * current_price)
    
    # Calculate metrics
    metrics_ml = SimpleBacktester.compute_metrics(portfolio_values_ml, 10000)
    metrics_ml["trades"] = buy_count
    
    logger.info("\nML Strategy Results:")
    logger.info(print_metrics_table(metrics_ml))
    
    # ===== COMPARISON =====
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 60)
    
    logger.info(f"\n{'Metric':<20} {'Rule-Based':<15} {'ML-Based':<15}")
    logger.info("-" * 50)
    logger.info(f"{'Total Return':<20} {metrics['total_return']:>13.2f}% {metrics_ml['total_return']:>13.2f}%")
    logger.info(f"{'Sharpe Ratio':<20} {metrics['sharpe_ratio']:>13.4f} {metrics_ml['sharpe_ratio']:>13.4f}")
    logger.info(f"{'Max Drawdown':<20} {metrics['max_drawdown']:>13.2f}% {metrics_ml['max_drawdown']:>13.2f}%")
    logger.info(f"{'Final Value':<20} ${metrics['final_value']:>12,.2f} ${metrics_ml['final_value']:>12,.2f}")
    logger.info(f"{'Trades':<20} {metrics['trades']:>13} {metrics_ml['trades']:>13}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example complete! Check README.md for more commands.")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        example_with_sample_data()
    except Exception as e:
        logger.error(f"Error running example: {str(e)}", exc_info=True)
        sys.exit(1)
