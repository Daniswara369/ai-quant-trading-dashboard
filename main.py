"""
AI Quant Trading System — Main CLI Entry Point

Usage:
    python main.py --mode train    --symbol BTCUSDT --timeframe 1h --model xgboost
    python main.py --mode signal   --symbol BTCUSDT --timeframe 1h --model xgboost
    python main.py --mode backtest --symbol BTCUSDT --timeframe 1h --model xgboost
    python main.py --mode realtime --symbols BTCUSDT ETHUSDT
"""
import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        █████╗ ██╗    ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗ ║
║       ██╔══██╗██║   ██╔═══██╗██║   ██║██╔══██╗████╗  ██║ ║
║       ███████║██║   ██║   ██║██║   ██║███████║██╗██╗ ██║  ║
║       ██╔══██║██║   ██║▀▄ ██║██║   ██║██╔══██║██║╚██╗██║ ║
║       ██║  ██║██║   ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║ ║
║       ╚═╝  ╚═╝╚═╝    ╚══▀══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝║
║                                                           ║
║           AI Quant Trading System v1.0                    ║
║           ML-Powered Market Analysis Platform             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="AI Quant Trading System — Main Entry Point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "signal", "backtest", "realtime"],
        help=(
            "Operating mode:\n"
            "  train     — Train ML model\n"
            "  signal    — Generate trading signals\n"
            "  backtest  — Run backtesting\n"
            "  realtime  — Start real-time monitoring"
        ),
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--symbols", nargs="+", default=None, help="Multiple symbols (for realtime)")
    parser.add_argument("--market", type=str, default=None, help="Market type: crypto/stock/forex")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "random_forest", "lightgbm", "lstm"])
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital for backtest")
    
    args = parser.parse_args()
    
    print(BANNER)
    
    if args.mode == "train":
        from data.data_pipeline import fetch_data
        from feature_engineering import engineer_features
        from models.model_trainer import train_pipeline
        from models.feature_importance import get_feature_importance, print_top_features
        
        print(f"  Mode    : TRAIN")
        print(f"  Symbol  : {args.symbol}")
        print(f"  Model   : {args.model}")
        print(f"  Timeframe: {args.timeframe}")
        print(f"  Tuning  : {'Yes' if args.tune else 'No'}")
        
        df = fetch_data(args.symbol, args.market, args.timeframe)
        if df.empty:
            print("[ERROR] No data fetched.")
            sys.exit(1)
        
        df = engineer_features(df)
        result = train_pipeline(df, args.symbol, args.model, args.timeframe, args.tune)
        
        if args.model != "lstm":
            imp = get_feature_importance(result["model"], result["feature_columns"], args.model)
            print_top_features(imp)
        
        print("\n  ✓ Training complete!")
    
    elif args.mode == "signal":
        from signal_generator import generate_signals
        from strategies.signal_rules import print_signal_summary, filter_actionable_signals
        
        print(f"  Mode    : SIGNAL GENERATION")
        print(f"  Symbol  : {args.symbol}")
        
        signals = generate_signals(args.symbol, args.market, args.timeframe, args.model)
        if not signals.empty:
            print_signal_summary(signals)
            actionable = filter_actionable_signals(signals)
            print(f"\nLast 10 actionable signals:")
            print(actionable.tail(10).to_string())
    
    elif args.mode == "backtest":
        from data.data_pipeline import fetch_data, detect_market_type
        from feature_engineering import engineer_features
        from models.model_trainer import load_model
        from strategies.signal_rules import generate_signal_from_probability
        from backtesting.backtest_engine import BacktestEngine
        from config import LSTM_PARAMS
        import numpy as np
        import pandas as pd
        
        market = args.market or detect_market_type(args.symbol)
        print(f"  Mode    : BACKTEST")
        print(f"  Symbol  : {args.symbol}")
        print(f"  Capital : ${args.capital:,.2f}")
        
        model, scaler, meta = load_model(args.symbol, args.model, args.timeframe)
        feature_cols = meta["feature_columns"]
        
        df = fetch_data(args.symbol, market, args.timeframe)
        df = engineer_features(df)
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        if args.model == "lstm":
            seq_len = LSTM_PARAMS["sequence_length"]
            Xs = [X_scaled[i - seq_len:i] for i in range(seq_len, len(X_scaled))]
            probs = model.predict(np.array(Xs), verbose=0).flatten()
            df = df.iloc[seq_len:]
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        
        signals = pd.Series(
            [generate_signal_from_probability(p) for p in probs],
            index=df.index[:len(probs)],
        )
        
        engine = BacktestEngine(initial_capital=args.capital, market_type=market)
        metrics = engine.run(df, signals)
        engine.print_metrics(metrics)
    
    elif args.mode == "realtime":
        from realtime.realtime_monitor import RealtimeMonitor
        from data.data_pipeline import detect_market_type
        
        symbols = args.symbols or [args.symbol]
        market_types = {s: detect_market_type(s) for s in symbols}
        
        print(f"  Mode    : REAL-TIME MONITORING")
        print(f"  Symbols : {', '.join(symbols)}")
        
        monitor = RealtimeMonitor(
            symbols=symbols,
            market_types=market_types,
            timeframe=args.timeframe,
            model_type=args.model,
        )
        monitor.run_loop()


if __name__ == "__main__":
    main()
