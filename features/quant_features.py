"""
Derived quantitative features for ML model input.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROLLING_VOL_WINDOW, MOMENTUM_WINDOW, SMA_PERIODS


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Logarithmic returns."""
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_rolling_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling window volatility (std of log returns)."""
    if "Log_Return" not in df.columns:
        df = add_log_returns(df)
    df["Rolling_Volatility"] = df["Log_Return"].rolling(window=ROLLING_VOL_WINDOW).std()
    return df


def add_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Price momentum over configurable window."""
    df["Momentum"] = df["Close"] / df["Close"].shift(MOMENTUM_WINDOW) - 1
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    return df


def add_distance_from_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Distance of current price from key moving averages (as %)."""
    for period in SMA_PERIODS:
        col = f"SMA_{period}"
        if col in df.columns:
            df[f"Dist_SMA_{period}"] = (df["Close"] - df[col]) / df[col]
    return df


def add_price_breakout_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price breakout signals:
    - Above Bollinger Upper Band
    - Below Bollinger Lower Band
    - Golden / Death cross (SMA 10 vs SMA 50)
    """
    # Bollinger breakout
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        df["BB_Breakout_Up"] = (df["Close"] > df["BB_Upper"]).astype(int)
        df["BB_Breakout_Down"] = (df["Close"] < df["BB_Lower"]).astype(int)
    
    # SMA cross
    if "SMA_10" in df.columns and "SMA_50" in df.columns:
        df["Golden_Cross"] = (
            (df["SMA_10"] > df["SMA_50"]) &
            (df["SMA_10"].shift(1) <= df["SMA_50"].shift(1))
        ).astype(int)
        df["Death_Cross"] = (
            (df["SMA_10"] < df["SMA_50"]) &
            (df["SMA_10"].shift(1) >= df["SMA_50"].shift(1))
        ).astype(int)
    
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candle-body features."""
    df["Body_Size"] = abs(df["Close"] - df["Open"]) / df["Open"]
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]
    df["Is_Bullish"] = (df["Close"] > df["Open"]).astype(int)
    return df


def add_lagged_features(df: pd.DataFrame, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    """Lagged close returns."""
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df["Close"].pct_change(lag)
    return df


def add_all_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all derived quantitative features.
    Should be called AFTER technical indicators are added.
    """
    df = df.copy()
    
    df = add_log_returns(df)
    df = add_rolling_volatility(df)
    df = add_momentum_signals(df)
    df = add_distance_from_ma(df)
    df = add_price_breakout_signals(df)
    df = add_candle_features(df)
    df = add_lagged_features(df)
    
    return df
