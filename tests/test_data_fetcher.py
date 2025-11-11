"""Unit tests for data fetcher module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data_fetcher import DataFetcher


class TestDataFetcher:
    """Test DataFetcher class."""
    
    def test_preprocess_data_shape(self):
        """Test that preprocessing returns correct shape."""
        # Create sample data
        df = pd.DataFrame({
            "open": np.random.randn(300) + 100,
            "high": np.random.randn(300) + 101,
            "low": np.random.randn(300) + 99,
            "close": np.random.randn(300) + 100,
            "volume": np.random.randint(1000000, 10000000, 300),
        })
        
        processed = DataFetcher.preprocess_data(df)
        
        # Should have original cols + label + 56 lagged features (4 cols * 14 lags)
        assert "label" in processed.columns
        assert any(col.endswith("_lag1") for col in processed.columns)
        assert len(processed) > 0  # Should have rows after dropping NaN
        
    def test_preprocess_data_label_validity(self):
        """Test that label is correctly calculated."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 102, 101],
            "volume": [1000000, 1000000, 1000000],
        })
        
        processed = DataFetcher.preprocess_data(df)
        
        # Label should be 1 if next close > current close
        # First row: next close (102) > current (100) = True = 1
        assert processed["label"].iloc[0] == 1
    
    def test_preprocess_data_no_nans(self):
        """Test that preprocessing removes NaN values."""
        df = pd.DataFrame({
            "open": [100, np.nan, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 102, 101],
            "volume": [1000000, 1000000, 1000000],
        })
        
        processed = DataFetcher.preprocess_data(df)
        
        # Should have no NaN values in feature columns
        assert not processed.isnull().any().any()
    
    def test_load_data(self):
        """Test loading data from CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample CSV
            df = pd.DataFrame({
                "close": [100, 101, 102],
                "volume": [1000000, 1000000, 1000000],
            })
            csv_path = os.path.join(tmpdir, "test.csv")
            df.to_csv(csv_path, index=False)
            
            # Load it
            loaded = DataFetcher.load_data(csv_path)
            
            assert len(loaded) == 3
            assert "close" in loaded.columns
            assert "volume" in loaded.columns


class TestSMACrossover:
    """Test SMA crossover calculations."""
    
    def test_sma_calculation(self):
        """Test that SMA is calculated correctly."""
        df = pd.DataFrame({
            "close": list(range(1, 101)),  # 1 to 100
        })
        
        sma_short = df["close"].rolling(10).mean()
        
        # First 9 values should be NaN
        assert sma_short.iloc[:9].isnull().all()
        
        # 10th value should be mean of 1-10 = 5.5
        assert sma_short.iloc[9] == 5.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
