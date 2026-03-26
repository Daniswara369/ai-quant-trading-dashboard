"""
Professional technical indicators for trading analysis.
Uses the `ta` library backed by pandas.
"""
import pandas as pd
import numpy as np
import ta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD, BOLLINGER_PERIOD,
    ATR_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCHASTIC_PERIOD, VOLUME_MA_PERIOD,
)


def add_sma(df: pd.DataFrame) -> pd.DataFrame:
    """Simple Moving Averages."""
    for period in SMA_PERIODS:
        df[f"SMA_{period}"] = ta.trend.sma_indicator(df["Close"], window=period)
    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    """Exponential Moving Averages."""
    for period in EMA_PERIODS:
        df[f"EMA_{period}"] = ta.trend.ema_indicator(df["Close"], window=period)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)."""
    macd = ta.trend.MACD(
        df["Close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()
    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Relative Strength Index."""
    df["RSI"] = ta.momentum.rsi(df["Close"], window=RSI_PERIOD)
    return df


def add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic Oscillator."""
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"], window=STOCHASTIC_PERIOD
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands."""
    bb = ta.volatility.BollingerBands(df["Close"], window=BOLLINGER_PERIOD)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Pct"] = bb.bollinger_pband()
    return df


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """Average True Range."""
    df["ATR"] = ta.volatility.average_true_range(
        df["High"], df["Low"], df["Close"], window=ATR_PERIOD
    )
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price (approximation)."""
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_tp_vol = (typical_price * df["Volume"]).cumsum()
        cum_vol = df["Volume"].cumsum()
        df["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    else:
        df["VWAP"] = (df["High"] + df["Low"] + df["Close"]) / 3
    return df


def add_volume_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Moving Average."""
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Volume_MA"] = df["Volume"].rolling(window=VOLUME_MA_PERIOD).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"].replace(0, np.nan)
    else:
        df["Volume_MA"] = 0.0
        df["Volume_Ratio"] = 1.0
    return df


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all technical indicators to an OHLCV DataFrame.
    """
    df = df.copy()
    
    df = add_sma(df)
    df = add_ema(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_vwap(df)
    df = add_volume_ma(df)
    
    return df
