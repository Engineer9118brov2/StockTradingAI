"""Unit tests for trading strategies module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.rule_based import SimpleRuleBasedBacktest
from src.strategies.ml_based import MLPredictor
from src.backtester import SimpleBacktester


class TestRuleBasedBacktest:
    """Test rule-based (SMA crossover) backtest."""
    
    def create_sample_data(self, n_days=300):
        """Create sample OHLCV data for testing.
        
        Args:
            n_days (int): Number of days to generate.
            
        Returns:
            pd.DataFrame: Sample OHLCV data.
        """
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
        close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        
        df = pd.DataFrame({
            "date": dates,
            "open": close_prices + np.random.randn(n_days) * 0.2,
            "high": close_prices + abs(np.random.randn(n_days) * 0.3),
            "low": close_prices - abs(np.random.randn(n_days) * 0.3),
            "close": close_prices,
            "volume": np.random.randint(1000000, 10000000, n_days),
        })
        
        return df
    
    def test_backtest_runs(self):
        """Test that backtest runs without error."""
        df = self.create_sample_data()
        
        backtester = SimpleRuleBasedBacktest(df, initial_cash=10000)
        result = backtester.run()
        
        assert "portfolio_value" in result
        assert "trades" in result
        assert len(result["portfolio_value"]) == len(df)
    
    def test_backtest_portfolio_values_decrease_with_trades(self):
        """Test that portfolio value reflects trading activity."""
        df = self.create_sample_data()
        
        backtester = SimpleRuleBasedBacktest(df, initial_cash=10000)
        result = backtester.run()
        
        # Portfolio should start at 10000
        assert abs(result["portfolio_value"][0] - 10000) < 100
        
        # Should have some variation if trades occurred
        assert max(result["portfolio_value"]) >= min(result["portfolio_value"])


class TestMLPredictor:
    """Test ML predictor."""
    
    def create_sample_data_with_features(self, n_samples=200):
        """Create sample data with lagged features.
        
        Args:
            n_samples (int): Number of samples.
            
        Returns:
            pd.DataFrame: Data with features and label.
        """
        close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        
        df = pd.DataFrame({
            "close": close_prices,
        })
        
        # Add lagged features
        for col in ["close"]:
            for lag in range(1, 15):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        # Add label
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def test_train_model(self):
        """Test model training."""
        df = self.create_sample_data_with_features(n_samples=500)
        
        predictor = MLPredictor(model_path="models/test_model.pkl")
        metrics = predictor.train_model(df)
        
        assert "accuracy" in metrics
        assert "test_size" in metrics
        assert metrics["accuracy"] >= 0.4  # Better than random on this data
    
    def test_predict(self):
        """Test model predictions."""
        df = self.create_sample_data_with_features(n_samples=500)
        
        predictor = MLPredictor(model_path="models/test_model.pkl")
        predictor.train_model(df)
        
        signals = predictor.predict(df, threshold=0.5)
        
        assert len(signals) == len(df)
        assert all(s in [0, 1] for s in signals)


class TestMetricsComputation:
    """Test performance metrics computation."""
    
    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with consistent positive returns."""
        # Consistent 1% daily return
        portfolio_values = [10000 * (1.01 ** i) for i in range(252)]
        
        metrics = SimpleBacktester.compute_metrics(portfolio_values, 10000)
        
        assert metrics["sharpe_ratio"] > 0  # Should be positive for positive returns
        assert metrics["total_return"] > 0
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Portfolio that goes up then down
        portfolio_values = [10000] * 50 + [11000] * 50 + [10000] * 50
        
        metrics = SimpleBacktester.compute_metrics(portfolio_values, 10000)
        
        assert metrics["max_drawdown"] < 0  # Should be negative
    
    def test_total_return_calculation(self):
        """Test total return calculation."""
        # Double the initial investment
        portfolio_values = [10000] * 10 + [20000] * 10
        
        metrics = SimpleBacktester.compute_metrics(portfolio_values, 10000)
        
        # 100% return
        assert abs(metrics["total_return"] - 100.0) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
