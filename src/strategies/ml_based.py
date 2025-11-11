"""ML-based trading strategy with support for multiple models.

This module implements ML-based strategies that predict whether the next day's
close price will be higher than the current close. Supports:
- Logistic Regression (simple baseline)
- RandomForestClassifier (improved, non-linear patterns)

Features can be basic lagged OHLC or enhanced with technical indicators
(RSI, MACD, Bollinger Bands via technical_indicators module).
"""

import os
import logging
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    import backtrader as bt
except ImportError:
    bt = None

logger = logging.getLogger(__name__)


class MLPredictor:
    """ML-based predictor supporting multiple models.
    
    Predicts price direction (up/down) based on lagged OHLC features.
    Supports both Logistic Regression and RandomForest models.
    
    Attributes:
        model: Trained ML model (LogisticRegression or RandomForestClassifier).
        scaler (StandardScaler): Feature scaler.
        feature_cols (List[str]): Names of feature columns.
        model_path (str): Path to save/load the model.
        model_type (str): Type of model ('logistic_regression' or 'random_forest').
    """
    
    def __init__(
        self,
        model_path: str = "models/ml_model.pkl",
        model_type: str = "random_forest",
    ):
        """Initialize ML predictor.
        
        Args:
            model_path (str): Path to save/load model files.
            model_type (str): Type of model to use.
                Options: 'logistic_regression', 'random_forest' (default).
        """
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_path = model_path
        self.scaler_path = model_path.replace(".pkl", "_scaler.pkl")
        self.features_path = model_path.replace(".pkl", "_features.pkl")
        self.model_type = model_type.lower()
        
        if self.model_type not in ["logistic_regression", "random_forest"]:
            logger.warning(f"Unknown model type '{model_type}'. Defaulting to 'random_forest'.")
            self.model_type = "random_forest"
    
    def train_model(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        """Train ML model on historical data.
        
        Args:
            df (pd.DataFrame): DataFrame with features and 'label' column.
                Expects columns like: close_lag1, close_lag2, ..., label
                Also supports TA features: rsi_lag*, macd_lag*, bb_* etc.
            test_size (float): Fraction of data for testing (default: 0.2).
            random_state (int): Random seed for reproducibility.
            
        Returns:
            Dict: Training metrics including accuracy, test_size, and feature importances.
            
        Raises:
            ValueError: If required columns are missing.
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Identify feature columns (all columns except date, open, high, low, close, volume, label)
        exclude_cols = {"date", "open", "high", "low", "close", "volume", "label"}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols or "label" not in df.columns:
            raise ValueError(
                "DataFrame must contain lagged features (close_lag1, ...) and 'label' column"
            )
        
        self.feature_cols = feature_cols
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        # Prepare data
        X = df[feature_cols].values
        y = df["label"].values
        
        # Drop any remaining NaNs
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Training set size: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )
            # RandomForest doesn't require scaling
            self.model.fit(X_train, y_train)
            X_train_eval = X_train
            X_test_eval = X_test
        else:  # logistic_regression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )
            self.model.fit(X_train_scaled, y_train)
            X_train_eval = X_train_scaled
            X_test_eval = X_test_scaled
        
        # Evaluate
        train_accuracy = self.model.score(X_train_eval, y_train)
        test_accuracy = self.model.score(X_test_eval, y_test)
        
        logger.info(f"Train accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Log feature importances (RandomForest only)
        if self.model_type == "random_forest" and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            logger.info("Top 10 features:")
            for idx in top_indices:
                logger.info(f"  {feature_cols[idx]}: {importances[idx]:.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_cols, self.features_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return {
            "accuracy": test_accuracy,
            "test_size": len(X_test),
            "train_accuracy": train_accuracy,
            "model_type": self.model_type,
        }
    
    def load_model(self):
        """Load pre-trained model and scaler.
        
        Raises:
            FileNotFoundError: If model files don't exist.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_cols = joblib.load(self.features_path)
        logger.info(f"Model loaded from {self.model_path}")
    
    def predict(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict buy/sell signals based on features.
        
        Args:
            df (pd.DataFrame): DataFrame with lagged features.
            threshold (float): Probability threshold for buy signal (default: 0.5).
                - > threshold: Buy signal (1)
                - <= threshold: Hold/Sell signal (0)
            
        Returns:
            np.ndarray: Signals array (1 = buy, 0 = hold/sell).
            
        Raises:
            RuntimeError: If model is not trained.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model() or load_model() first.")
        
        if self.feature_cols is None:
            raise RuntimeError("Feature columns not set. Train model first.")
        
        # Get features (may not exist for all rows, especially at beginning of time series)
        X = df[self.feature_cols].values
        
        # Handle missing values
        mask = ~np.isnan(X).any(axis=1)
        signals = np.zeros(len(X))
        
        if mask.sum() > 0:
            X_valid = X[mask]
            
            # Scale only for Logistic Regression
            if self.model_type == "random_forest":
                proba = self.model.predict_proba(X_valid)[:, 1]
            else:
                X_valid_scaled = self.scaler.transform(X_valid)
                proba = self.model.predict_proba(X_valid_scaled)[:, 1]
            
            signals[mask] = (proba > threshold).astype(int)
        
        return signals
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get probability predictions.
        
        Args:
            df (pd.DataFrame): DataFrame with lagged features.
            
        Returns:
            np.ndarray: Probability of buy signal for each row.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained.")
        
        X = df[self.feature_cols].values
        mask = ~np.isnan(X).any(axis=1)
        proba = np.zeros(len(X))
        
        if mask.sum() > 0:
            X_valid = X[mask]
            X_scaled = self.scaler.transform(X_valid)
            proba[mask] = self.model.predict_proba(X_scaled)[:, 1]
        
        return proba


class MLStrategy(bt.Strategy if bt else object):
    """ML-based trading strategy using backtrader.
    
    Uses pre-trained ML model to generate buy/sell signals.
    
    Params:
        ml_predictor (MLPredictor): Trained ML predictor.
        trade_size (int): Number of shares per trade (default: 10).
    """
    
    params = {
        "ml_predictor": None,
        "trade_size": 10,
    }
    
    def __init__(self):
        """Initialize the strategy."""
        if bt is None:
            raise RuntimeError("backtrader not installed.")
        
        if self.params.ml_predictor is None:
            raise ValueError("ml_predictor parameter must be set.")
        
        self.trades = []
        self.position_size = 0
    
    def next(self):
        """Execute trading logic based on ML predictions."""
        # Get current bar index
        idx = len(self) - 1
        
        # This would require passing signals, simplified for demo
        logger.warning("MLStrategy requires external signal injection")
