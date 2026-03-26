"""
Entry point for the real-time signal engine.

Usage:
    python realtime_engine.py
    python realtime_engine.py --symbols BTCUSDT ETHUSDT --timeframe 1h --model xgboost
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime.realtime_monitor import RealtimeMonitor
from data.data_pipeline import detect_market_type
from config import CRYPTO_ASSETS, REALTIME_REFRESH_SECONDS


def main():
    parser = argparse.ArgumentParser(description="Real-time signal engine")
    parser.add_argument("--symbols", nargs="+", default=CRYPTO_ASSETS,
                        help="Symbols to monitor")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--refresh", type=int, default=REALTIME_REFRESH_SECONDS,
                        help="Refresh interval in seconds")
    args = parser.parse_args()
    
    market_types = {s: detect_market_type(s) for s in args.symbols}
    
    monitor = RealtimeMonitor(
        symbols=args.symbols,
        market_types=market_types,
        timeframe=args.timeframe,
        model_type=args.model,
        refresh_seconds=args.refresh,
    )
    
    monitor.run_loop()


if __name__ == "__main__":
    main()
