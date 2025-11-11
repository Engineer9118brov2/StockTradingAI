"""Data fetching and preprocessing module for stock data.

This module provides functionality to fetch historical stock data from the Polygon API
and preprocess it for trading strategies.
"""

import os
import logging
from typing import Optional
import pandas as pd
import numpy as np

try:
    from polygon import RESTClient
except ImportError:
    RESTClient = None

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and preprocesses stock data from Polygon API.
    
    Attributes:
        api_key (str): Polygon API key from environment variable.
        client (RESTClient): Polygon API client instance.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DataFetcher with API key.
        
        Args:
            api_key (Optional[str]): Polygon API key. If None, reads from POLYGON_API_KEY env var.
            
        Raises:
            ValueError: If no API key is provided and POLYGON_API_KEY env var is not set.
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key and RESTClient is not None:
            raise ValueError(
                "API key not provided. Set POLYGON_API_KEY environment variable or pass api_key parameter."
            )
        
        if RESTClient is not None:
            self.client = RESTClient(self.api_key)
        else:
            self.client = None
            logger.warning("polygon-api-client not installed. Some features may be unavailable.")
    
    def fetch_data(
        self,
        ticker: str = "AAPL",
        days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol (default: 'AAPL').
            days (int): Number of days of historical data to fetch (default: 365).
            start_date (Optional[str]): Start date in 'YYYY-MM-DD' format. Overrides days if provided.
            end_date (Optional[str]): End date in 'YYYY-MM-DD' format.
            
        Returns:
            pd.DataFrame: DataFrame with columns [date, open, high, low, close, volume]
            
        Raises:
            RuntimeError: If API client is not available or API call fails.
        """
        if self.client is None:
            logger.error("Polygon client not available. Install polygon-api-client.")
            raise RuntimeError("Polygon client not initialized. Install polygon-api-client.")
        
        try:
            logger.info(f"Fetching {days} days of data for {ticker}...")
            
            # Calculate date range if not provided
            from datetime import datetime, timedelta
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)
                start_date = start_dt.strftime("%Y-%m-%d")
            
            # Fetch aggregates (daily bars) from Polygon API
            aggs = []
            for agg in self.client.list_aggs(
                ticker=ticker,
                timespan="day",
                multiplier=1,
                from_=start_date,
                to=end_date,
                limit=50000,
            ):
                aggs.append(agg)
            
            if not aggs:
                raise RuntimeError(f"No data fetched for ticker {ticker}")
            
            # Convert to DataFrame
            data = pd.DataFrame([
                {
                    "date": pd.to_datetime(agg.timestamp),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                }
                for agg in aggs
            ])
            
            # Sort by date and drop duplicates
            data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            data["date"] = pd.to_datetime(data["date"]).dt.date
            
            logger.info(f"Fetched {len(data)} records for {ticker}")
            
            # Save to CSV
            os.makedirs("data", exist_ok=True)
            csv_path = f"data/{ticker}_data.csv"
            data.to_csv(csv_path, index=False)
            logger.info(f"Data saved to {csv_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise RuntimeError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data: add lagged features and labels for ML.
        
        Args:
            df (pd.DataFrame): Raw OHLCV DataFrame with columns [open, high, low, close, volume].
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with added features:
                - label: 1 if next close > current close, else 0
                - close_lag1, close_lag2, ..., close_lag14
                - open_lag1, ..., open_lag14, etc.
        """
        df = df.copy()
        
        # Drop rows with NaN
        df = df.dropna()
        
        # Add label for ML (1 if next day close > today close)
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
        
        # Add lagged features (14 lags for each OHLC)
        for col in ["close", "open", "high", "low"]:
            for lag in range(1, 15):
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        
        # Drop first and last rows (due to shifting)
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """Load preprocessed data from CSV.
        
        Args:
            filepath (str): Path to CSV file.
            
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
