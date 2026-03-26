"""
Fetch forex OHLCV data.
Primary: Yahoo Finance   Fallback: Alpha Vantage / TwelveData
"""
import pandas as pd
import yfinance as yf
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FOREX_YAHOO_MAP,
    YFINANCE_INTERVAL_MAP,
    YFINANCE_PERIOD_MAP,
    ALPHA_VANTAGE_API_KEY,
    TWELVE_DATA_API_KEY,
)


def fetch_forex_yahoo(symbol: str, timeframe: str = "1d", period: str = None) -> pd.DataFrame:
    """
    Fetch forex data via Yahoo Finance (e.g. EURUSD → EURUSD=X).
    """
    yahoo_ticker = FOREX_YAHOO_MAP.get(symbol.upper(), f"{symbol.upper()}=X")
    interval = YFINANCE_INTERVAL_MAP.get(timeframe, "1d")
    if period is None:
        period = YFINANCE_PERIOD_MAP.get(timeframe, "max")
    
    try:
        ticker = yf.Ticker(yahoo_ticker)
        df = ticker.history(period=period, interval=interval)
    except Exception as e:
        print(f"[ERROR] Yahoo Finance forex fetch failed for {symbol}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        print(f"[WARNING] No Yahoo Finance data for {symbol}")
        return pd.DataFrame()
    
    df.index.name = "DateTime"
    
    cols_to_keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols_to_keep]
    
    for col in cols_to_keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    df.sort_index(inplace=True)
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    
    return df


def fetch_forex_alpha_vantage(symbol: str, timeframe: str = "1d") -> pd.DataFrame:
    """
    Fetch forex data from Alpha Vantage (requires API key).
    """
    if not ALPHA_VANTAGE_API_KEY:
        print("[WARNING] No Alpha Vantage API key configured.")
        return pd.DataFrame()
    
    from_symbol = symbol[:3].upper()
    to_symbol = symbol[3:].upper()
    
    function_map = {
        "1m": "FX_INTRADAY",
        "5m": "FX_INTRADAY",
        "15m": "FX_INTRADAY",
        "1h": "FX_INTRADAY",
        "1d": "FX_DAILY",
    }
    
    func = function_map.get(timeframe, "FX_DAILY")
    
    params = {
        "function": func,
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "full",
    }
    
    if func == "FX_INTRADAY":
        av_interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "60min"}
        params["interval"] = av_interval_map.get(timeframe, "60min")
    
    try:
        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Alpha Vantage fetch failed: {e}")
        return pd.DataFrame()
    
    # Find the time series key
    ts_key = None
    for key in data:
        if "Time Series" in key:
            ts_key = key
            break
    
    if ts_key is None:
        print(f"[WARNING] No time series data in Alpha Vantage response for {symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(data[ts_key], orient="index")
    df.index = pd.to_datetime(df.index)
    df.index.name = "DateTime"
    
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if "open" in cl:
            rename[col] = "Open"
        elif "high" in cl:
            rename[col] = "High"
        elif "low" in cl:
            rename[col] = "Low"
        elif "close" in cl:
            rename[col] = "Close"
    
    df.rename(columns=rename, inplace=True)
    df = df[["Open", "High", "Low", "Close"]]
    df["Volume"] = 0.0  # Forex doesn't have centralized volume here
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.sort_index(inplace=True)
    
    return df


def fetch_forex_data(symbol: str, timeframe: str = "1d", period: str = None) -> pd.DataFrame:
    """
    Fetch forex data. Try Yahoo Finance first, fallback to Alpha Vantage.
    """
    df = fetch_forex_yahoo(symbol, timeframe, period)
    
    if df.empty:
        print(f"[INFO] Trying Alpha Vantage fallback for {symbol}...")
        df = fetch_forex_alpha_vantage(symbol, timeframe)
    
    return df
