"""
CLI script to train ML models for price direction prediction.

Usage:
    python train_model.py --symbol BTCUSDT --market crypto --timeframe 1h --model xgboost
    python train_model.py --symbol AAPL --market stock --timeframe 1d --model lightgbm --tune
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import fetch_data
from feature_engineering import engineer_features
from models.model_trainer import train_pipeline
from models.feature_importance import get_feature_importance, print_top_features


def main():
    parser = argparse.ArgumentParser(description="Train ML model for price prediction")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--market", type=str, default=None, help="Market type: crypto/stock/forex")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "random_forest", "lightgbm", "lstm"],
                        help="Model type")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    args = parser.parse_args()
    
    print(f"\n{'═'*60}")
    print(f"  AI QUANT TRADING SYSTEM — MODEL TRAINING")
    print(f"  Symbol: {args.symbol} | Market: {args.market or 'auto'}")
    print(f"  Timeframe: {args.timeframe} | Model: {args.model}")
    print(f"{'═'*60}")
    
    # Step 1: Fetch data
    print("\n[1/4] Fetching market data...")
    df = fetch_data(args.symbol, args.market, args.timeframe)
    if df.empty:
        print("[ERROR] No data fetched. Exiting.")
        sys.exit(1)
    print(f"  Loaded {len(df)} candles")
    
    # Step 2: Feature engineering
    print("\n[2/4] Engineering features...")
    df = engineer_features(df)
    print(f"  Feature-enriched data: {df.shape}")
    
    # Step 3: Train model
    print("\n[3/4] Training model...")
    result = train_pipeline(df, args.symbol, args.model, args.timeframe, tune=args.tune)
    
    # Step 4: Feature importance
    if args.model != "lstm":
        print("\n[4/4] Feature importance analysis...")
        imp_df = get_feature_importance(result["model"], result["feature_columns"], args.model)
        print_top_features(imp_df)
    
    print(f"\n{'═'*60}")
    print(f"  ✓ Training complete!")
    print(f"  Model saved at: {result['model_path']}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
