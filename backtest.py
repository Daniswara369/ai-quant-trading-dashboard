"""
CLI script to run backtesting on historical data.

Usage:
    python backtest.py --symbol BTCUSDT --market crypto --timeframe 1h --model xgboost
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import fetch_data, detect_market_type
from feature_engineering import engineer_features
from models.model_trainer import load_model, create_target
from strategies.signal_rules import generate_signal_from_probability
from backtesting.backtest_engine import BacktestEngine
from config import INITIAL_CAPITAL, LSTM_PARAMS
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--market", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()
    
    market = args.market or detect_market_type(args.symbol)
    
    print(f"\n{'═'*60}")
    print(f"  AI QUANT TRADING SYSTEM — BACKTESTING")
    print(f"  Symbol: {args.symbol} | Market: {market}")
    print(f"  Timeframe: {args.timeframe} | Model: {args.model}")
    print(f"  Capital: ${args.capital:,.2f}")
    print(f"{'═'*60}")
    
    # Load model
    print("\n[1/4] Loading model...")
    try:
        model, scaler, meta = load_model(args.symbol, args.model, args.timeframe)
    except FileNotFoundError:
        print(f"[ERROR] No trained model found. Train first:")
        print(f"  python train_model.py --symbol {args.symbol} --timeframe {args.timeframe} --model {args.model}")
        sys.exit(1)
    
    feature_cols = meta["feature_columns"]
    
    # Fetch data
    print("\n[2/4] Fetching data...")
    df = fetch_data(args.symbol, market, args.timeframe)
    if df.empty:
        print("[ERROR] No data. Exiting.")
        sys.exit(1)
    
    # Engineer features
    print("\n[3/4] Engineering features...")
    df = engineer_features(df)
    
    # Fill missing features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Generate predictions → signals
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    if args.model == "lstm":
        seq_len = LSTM_PARAMS["sequence_length"]
        Xs = []
        for i in range(seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - seq_len:i])
        Xs = np.array(Xs)
        probabilities = model.predict(Xs, verbose=0).flatten()
        df = df.iloc[seq_len:]
        probabilities_full = probabilities
    else:
        probabilities = model.predict_proba(X_scaled)[:, 1]
        probabilities_full = probabilities
    
    signals = pd.Series(
        [generate_signal_from_probability(p) for p in probabilities_full],
        index=df.index[:len(probabilities_full)]
    )
    
    # Run backtest
    print("\n[4/4] Running backtest...")
    engine = BacktestEngine(
        initial_capital=args.capital,
        market_type=market,
    )
    metrics = engine.run(df, signals)
    engine.print_metrics(metrics)
    
    # Trade summary
    trades_df = engine.get_trades_df()
    if not trades_df.empty:
        print(f"\nLast 10 trades:")
        print(trades_df.tail(10).to_string())
    
    print(f"\n{'═'*60}")
    print(f"  ✓ Backtest complete!")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
