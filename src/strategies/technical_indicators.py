"""Technical indicators for enhanced feature engineering.

This module provides functions to calculate common technical indicators
(RSI, MACD, Bollinger Bands) and add them to OHLCV data with lagging
for ML model training.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_rsi(df: pd.DataFrame, window: int = 14, column: str = "close") -> pd.DataFrame:
    """Add Relative Strength Index (RSI) indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        window (int): RSI period (default: 14).
        column (str): Price column to use (default: 'close').
        
    Returns:
        pd.DataFrame: DataFrame with 'rsi' column added.
    """
    df = df.copy()
    
    # Calculate price changes
    delta = df[column].diff()
    
    # Separate gains and losses
    gains = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    losses = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    
    df['rsi'] = rsi
    logger.debug(f"Added RSI({window}) to DataFrame")
    
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence) indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        fast (int): Fast EMA period (default: 12).
        slow (int): Slow EMA period (default: 26).
        signal (int): Signal line period (default: 9).
        column (str): Price column to use (default: 'close').
        
    Returns:
        pd.DataFrame: DataFrame with 'macd', 'macd_signal', 'macd_hist' columns added.
    """
    df = df.copy()
    
    # Calculate EMAs
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    # MACD line
    macd = ema_fast - ema_slow
    
    # Signal line
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    
    # Histogram
    macd_hist = macd - macd_signal
    
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    logger.debug(f"Added MACD({fast},{slow},{signal}) to DataFrame")
    
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    std_dev: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """Add Bollinger Bands indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        window (int): SMA period (default: 20).
        std_dev (float): Standard deviation multiplier (default: 2.0).
        column (str): Price column to use (default: 'close').
        
    Returns:
        pd.DataFrame: DataFrame with 'bb_upper', 'bb_middle', 'bb_lower' columns added.
    """
    df = df.copy()
    
    # Middle band (SMA)
    middle = df[column].rolling(window=window).mean()
    
    # Standard deviation
    std = df[column].rolling(window=window).std()
    
    # Upper and lower bands
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # Bollinger Bandwidth (useful feature)
    bandwidth = (upper - lower) / middle
    
    # Price position within bands (0-1 scale)
    bb_position = (df[column] - lower) / (upper - lower)
    
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_bandwidth'] = bandwidth
    df['bb_position'] = bb_position
    
    logger.debug(f"Added Bollinger Bands({window},{std_dev}) to DataFrame")
    
    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with volume indicators added.
    """
    df = df.copy()
    
    # Volume moving averages
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ma_50'] = df['volume'].rolling(window=50).mean()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    logger.debug("Added volume indicators to DataFrame")
    
    return df


def add_lagged_technical_features(
    df: pd.DataFrame,
    lags: int = 14,
    indicators: list = None,
) -> pd.DataFrame:
    """Add lagged technical indicator features for ML.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators already added.
        lags (int): Number of lags to create (default: 14).
        indicators (list): List of indicator columns to lag.
                          If None, lags RSI, MACD, and volume-related columns.
        
    Returns:
        pd.DataFrame: DataFrame with lagged indicator features added.
    """
    df = df.copy()
    
    if indicators is None:
        indicators = ['rsi', 'macd', 'macd_signal', 'bb_bandwidth', 'bb_position', 'volume_ratio']
    
    # Filter to only existing columns
    indicators = [col for col in indicators if col in df.columns]
    
    # Add lags for each indicator
    for indicator in indicators:
        for lag in range(1, lags + 1):
            df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)
    
    logger.debug(f"Added {len(indicators)} lagged technical features ({lags} lags each)")
    
    return df


def add_all_technical_features(df: pd.DataFrame, lags: int = 14) -> pd.DataFrame:
    """Add all technical indicators and their lags in one call.
    
    This is a convenience function that applies all TA indicators and their lags.
    Order: RSI → MACD → Bollinger Bands → Volume → Lags.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (requires 'close', 'volume').
        lags (int): Number of lags for each indicator (default: 14).
        
    Returns:
        pd.DataFrame: DataFrame with all technical features added.
    """
    logger.info("Adding all technical indicators...")
    
    df = add_rsi(df, window=14)
    df = add_macd(df, fast=12, slow=26, signal=9)
    df = add_bollinger_bands(df, window=20, std_dev=2.0)
    df = add_volume_indicators(df)
    df = add_lagged_technical_features(df, lags=lags)
    
    # Drop initial NaN rows from indicators
    df = df.dropna().reset_index(drop=True)
    
    logger.info(f"Technical features complete. Shape: {df.shape}")
    
    return df
