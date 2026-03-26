"""
Fetch stock market OHLCV data using yfinance.
"""
import pandas as pd
import yfinance as yf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YFINANCE_INTERVAL_MAP, YFINANCE_PERIOD_MAP


def fetch_stock_data(symbol: str, timeframe: str = "1d", period: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV data for a stock using yfinance.
    
    Args:
        symbol: Stock ticker, e.g. 'AAPL'
        timeframe: Candle interval ('1m', '5m', '15m', '1h', '1d')
        period: yfinance period string. If None, auto-selected based on timeframe.
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, DateTime index
    """
    interval = YFINANCE_INTERVAL_MAP.get(timeframe, "1d")
    if period is None:
        period = YFINANCE_PERIOD_MAP.get(timeframe, "max")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    except Exception as e:
        print(f"[ERROR] Failed to fetch stock data for {symbol}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        print(f"[WARNING] No data returned for {symbol}")
        return pd.DataFrame()
    
    # Standardize columns
    df.index.name = "DateTime"
    
    # Keep only OHLCV
    cols_to_keep = []
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            cols_to_keep.append(col)
    
    df = df[cols_to_keep]
    
    # Ensure numeric
    for col in cols_to_keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove timezone info for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    
    return df
