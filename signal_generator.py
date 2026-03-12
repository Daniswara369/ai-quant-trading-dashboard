"""
End-to-end signal generator.
Load model → fetch data → engineer features → predict → generate signals.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import fetch_data
from feature_engineering import engineer_features, get_feature_columns
from models.model_trainer import load_model, create_target
from strategies.signal_rules import (
    generate_signals_batch, filter_actionable_signals, print_signal_summary,
)
from config import LSTM_PARAMS


def generate_signals(
    symbol: str,
    market_type: str = None,
    timeframe: str = "1h",
    model_type: str = "xgboost",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Complete signal generation pipeline.
    
    Returns:
        DataFrame with columns: timestamp, symbol, entry_price, signal,
                                 probability_up, probability_down
    """
    # Load trained model
    try:
        model, scaler, meta = load_model(symbol, model_type, timeframe)
    except FileNotFoundError:
        print(f"[ERROR] No trained model found for {symbol}/{model_type}/{timeframe}.")
        print(f"  Run: python train_model.py --symbol {symbol} --timeframe {timeframe} --model {model_type}")
        return pd.DataFrame()
    
    feature_cols = meta["feature_columns"]
    
    # Fetch latest data
    df = fetch_data(symbol, market_type, timeframe, force_refresh=force_refresh)
    if df.empty:
        return pd.DataFrame()
    
    # Engineer features
    df = engineer_features(df)
    if df.empty:
        return pd.DataFrame()
    
    # Select and scale features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing features: {missing}")
        for col in missing:
            df[col] = 0.0
    
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Predict
    if model_type == "lstm":
        seq_len = LSTM_PARAMS["sequence_length"]
        if len(X_scaled) < seq_len:
            print(f"[ERROR] Not enough data for LSTM (need {seq_len}, have {len(X_scaled)})")
            return pd.DataFrame()
        
        Xs = []
        for i in range(seq_len, len(X_scaled)):
            Xs.append(X_scaled[i - seq_len:i])
        Xs = np.array(Xs)
        probabilities = model.predict(Xs, verbose=0).flatten()
        df_for_signals = df.iloc[seq_len:]
    else:
        probabilities = model.predict_proba(X_scaled)[:, 1]
        df_for_signals = df
    
    # Generate signals
    signals_df = generate_signals_batch(df_for_signals, probabilities, symbol)
    
    return signals_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading signals")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--market", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--model", type=str, default="xgboost")
    args = parser.parse_args()
    
    signals = generate_signals(args.symbol, args.market, args.timeframe, args.model)
    if not signals.empty:
        print_signal_summary(signals)
        actionable = filter_actionable_signals(signals)
        print(f"\nLast 10 actionable signals:")
        print(actionable.tail(10).to_string())
