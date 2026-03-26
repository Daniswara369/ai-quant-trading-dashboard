"""
Batch train XGBoost models for all dashboard assets and run backtests.
Saves backtest results as JSON to backtesting/results/.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import fetch_data, detect_market_type
from feature_engineering import engineer_features
from models.model_trainer import train_pipeline, load_model
from strategies.signal_rules import generate_signal_from_probability
from backtesting.backtest_engine import BacktestEngine
from config import MODEL_SAVE_DIR
import numpy as np
import pandas as pd

DASHBOARD_ASSETS = ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT", "TSLA", "EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "1h"
MODEL_TYPE = "xgboost"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtesting", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def model_exists(symbol):
    prefix = f"{symbol}_{MODEL_TYPE}_{TIMEFRAME}"
    return os.path.exists(os.path.join(MODEL_SAVE_DIR, f"{prefix}_model.joblib"))


def train_asset(symbol):
    print(f"\n{'='*60}")
    print(f"  TRAINING: {symbol}")
    print(f"{'='*60}")
    market = detect_market_type(symbol)
    df = fetch_data(symbol, market, TIMEFRAME)
    if df.empty:
        print(f"  [SKIP] No data for {symbol}")
        return False
    df = engineer_features(df)
    if len(df) < 100:
        print(f"  [SKIP] Insufficient data for {symbol} ({len(df)} rows)")
        return False
    train_pipeline(df, symbol, MODEL_TYPE, TIMEFRAME, tune=False)
    return True


def run_backtest(symbol):
    print(f"\n  Running backtest for {symbol}...")
    market = detect_market_type(symbol)
    try:
        model, scaler, meta = load_model(symbol, MODEL_TYPE, TIMEFRAME)
    except FileNotFoundError:
        print(f"  [SKIP] No model for {symbol}")
        return None

    feature_cols = meta["feature_columns"]
    df = fetch_data(symbol, market, TIMEFRAME)
    if df.empty:
        return None
    df = engineer_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    signals = pd.Series(
        [generate_signal_from_probability(p) for p in probs],
        index=df.index[:len(probs)],
    )

    engine = BacktestEngine(market_type=market)
    metrics = engine.run(df, signals)
    engine.print_metrics(metrics)

    # Save equity curve
    equity_df = engine.get_equity_curve_df(df)
    equity_data = [
        {"timestamp": str(t), "equity": float(e)}
        for t, e in zip(equity_df.index, equity_df["Equity"])
    ]

    result = {
        "symbol": symbol,
        "market": market,
        "timeframe": TIMEFRAME,
        "model": MODEL_TYPE,
        "metrics": metrics,
        "equity_curve": equity_data,
    }

    result_path = os.path.join(RESULTS_DIR, f"{symbol}_{TIMEFRAME}_backtest.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [SAVED] {result_path}")
    return metrics


def main():
    print("\n" + "=" * 60)
    print("  BATCH MODEL TRAINING & BACKTESTING")
    print("=" * 60)

    for symbol in DASHBOARD_ASSETS:
        if not model_exists(symbol):
            train_asset(symbol)
        else:
            print(f"\n  [EXISTS] Model for {symbol} already trained.")

    print("\n\n" + "=" * 60)
    print("  RUNNING BACKTESTS FOR ALL ASSETS")
    print("=" * 60)

    for symbol in DASHBOARD_ASSETS:
        run_backtest(symbol)

    print("\n\n  ALL DONE!")


if __name__ == "__main__":
    main()
