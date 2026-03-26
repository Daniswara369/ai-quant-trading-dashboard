"""
Fetch cryptocurrency OHLCV data from Binance public API.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BINANCE_INTERVAL_MAP

# Use the data-api endpoint to avoid 451 IP blocks from US-based cloud servers (Hugging Face)
BINANCE_BASE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_crypto_data(symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance for a given crypto pair.
    
    Args:
        symbol: Trading pair, e.g. 'BTCUSDT'
        timeframe: Candle interval ('1m', '5m', '15m', '1h', '1d')
        limit: Number of candles to fetch (max 1000)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, DateTime index
    """
    interval = BINANCE_INTERVAL_MAP.get(timeframe, "1h")
    
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000),
    }
    
    try:
        response = requests.get(BINANCE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch crypto data for {symbol}: {e}")
        return pd.DataFrame()
    
    if not data:
        print(f"[WARNING] No data returned for {symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ])
    
    # Convert types
    df["DateTime"] = pd.to_datetime(df["Open Time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    
    df = df[["DateTime", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("DateTime", inplace=True)
    df.sort_index(inplace=True)
    
    return df


def fetch_crypto_data_extended(symbol: str, timeframe: str = "1h", days: int = 90) -> pd.DataFrame:
    """
    Fetch extended crypto data by paginating through Binance API.
    """
    interval = BINANCE_INTERVAL_MAP.get(timeframe, "1h")
    
    # Calculate time intervals per candle in ms
    interval_ms_map = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "1d": 86_400_000,
    }
    interval_ms = interval_ms_map.get(timeframe, 3_600_000)
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }
        
        try:
            response = requests.get(BINANCE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Pagination error for {symbol}: {e}")
            break
        
        if not data:
            break
        
        all_data.extend(data)
        current_start = data[-1][0] + interval_ms
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ])
    
    df["DateTime"] = pd.to_datetime(df["Open Time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    
    df = df[["DateTime", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("DateTime", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    
    return df
