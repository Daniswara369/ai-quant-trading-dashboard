"""
Unified data pipeline — single entry point for fetching and caching market data.
"""
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CACHE_DIR, CRYPTO_ASSETS, STOCK_ASSETS, FOREX_ASSETS
from data.crypto_data import fetch_crypto_data, fetch_crypto_data_extended
from data.stock_data import fetch_stock_data
from data.forex_data import fetch_forex_data


def detect_market_type(symbol: str) -> str:
    """Auto-detect market type from symbol."""
    symbol_upper = symbol.upper()
    if symbol_upper in CRYPTO_ASSETS:
        return "crypto"
    elif symbol_upper in STOCK_ASSETS:
        return "stock"
    elif symbol_upper in FOREX_ASSETS:
        return "forex"
    # Heuristic fallback
    elif symbol_upper.endswith("USDT") or symbol_upper.endswith("BTC"):
        return "crypto"
    elif len(symbol_upper) == 6 and symbol_upper[:3].isalpha() and symbol_upper[3:].isalpha():
        return "forex"
    else:
        return "stock"


def _cache_path(symbol: str, market_type: str, timeframe: str) -> str:
    """Generate cache file path."""
    return os.path.join(DATA_CACHE_DIR, f"{symbol}_{market_type}_{timeframe}.csv")


def fetch_data(
    symbol: str,
    market_type: str = None,
    timeframe: str = "1h",
    use_cache: bool = True,
    force_refresh: bool = False,
    extended_days: int = 90,
) -> pd.DataFrame:
    """
    Unified data fetcher.
    
    Args:
        symbol: Asset symbol (BTCUSDT, AAPL, EURUSD, etc.)
        market_type: 'crypto', 'stock', 'forex'. Auto-detected if None.
        timeframe: '1m', '5m', '15m', '1h', '1d'
        use_cache: Whether to read/write CSV cache.
        force_refresh: Force re-download even if cache exists.
        extended_days: For crypto, how many days of history to fetch.
    
    Returns:
        OHLCV DataFrame with DateTime index.
    """
    if market_type is None:
        market_type = detect_market_type(symbol)
    
    cache_file = _cache_path(symbol, market_type, timeframe)
    
    # Try cache first
    if use_cache and not force_refresh and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col="DateTime", parse_dates=True)
            if not df.empty:
                print(f"[CACHE] Loaded {len(df)} rows for {symbol} ({timeframe})")
                return df
        except Exception:
            pass
    
    # Fetch fresh data
    print(f"[FETCH] Downloading {symbol} ({market_type}, {timeframe})...")
    
    if market_type == "crypto":
        df = fetch_crypto_data_extended(symbol, timeframe, days=extended_days)
    elif market_type == "stock":
        df = fetch_stock_data(symbol, timeframe)
    elif market_type == "forex":
        df = fetch_forex_data(symbol, timeframe)
    else:
        print(f"[ERROR] Unknown market type: {market_type}")
        return pd.DataFrame()
    
    # Save cache
    if use_cache and not df.empty:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        df.to_csv(cache_file)
        print(f"[CACHE] Saved {len(df)} rows to {cache_file}")
    
    return df


def fetch_all_assets(timeframe: str = "1h") -> dict:
    """
    Fetch data for all configured assets.
    
    Returns:
        Dictionary mapping symbol → DataFrame.
    """
    results = {}
    
    all_symbols = [
        ("crypto", CRYPTO_ASSETS),
        ("stock", STOCK_ASSETS),
        ("forex", FOREX_ASSETS),
    ]
    
    for market_type, symbols in all_symbols:
        for symbol in symbols:
            df = fetch_data(symbol, market_type, timeframe)
            if not df.empty:
                results[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} rows")
            else:
                print(f"  ✗ {symbol}: no data")
    
    return results


def update_data(symbol: str, market_type: str = None, timeframe: str = "1h") -> pd.DataFrame:
    """
    Update cached data by fetching fresh data and merging.
    """
    return fetch_data(symbol, market_type, timeframe, use_cache=True, force_refresh=True)


if __name__ == "__main__":
    # Quick test
    df = fetch_data("BTCUSDT", "crypto", "1h")
    print(f"\nBTCUSDT shape: {df.shape}")
    if not df.empty:
        print(df.tail())
