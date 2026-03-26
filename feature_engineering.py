"""
Feature Engineering Orchestrator.
Takes raw OHLCV → applies all indicators and quant features → returns enriched DataFrame.
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.technical_indicators import add_all_technical_indicators
from features.quant_features import add_all_quant_features


def engineer_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Args:
        df: Raw OHLCV DataFrame with DateTime index.
        drop_na: Whether to drop rows with NaN (from rolling calcs).
    
    Returns:
        Enriched DataFrame with all features.
    """
    if df.empty:
        print("[WARNING] Empty DataFrame passed to feature engineering.")
        return df
    
    # Step 1: Technical indicators
    df = add_all_technical_indicators(df)
    
    # Step 2: Quantitative features
    df = add_all_quant_features(df)
    
    # Step 3: Drop NaN rows from rolling calculations
    if drop_na:
        initial_len = len(df)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"[INFO] Dropped {dropped} rows with NaN values ({len(df)} remaining)")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature column names (excluding raw OHLCV and target).
    """
    exclude = {"Open", "High", "Low", "Close", "Volume", "Target", "DateTime"}
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    from data.data_pipeline import fetch_data
    
    df = fetch_data("BTCUSDT", "crypto", "1h")
    if not df.empty:
        fdf = engineer_features(df)
        print(f"\nFeature-enriched shape: {fdf.shape}")
        print(f"Feature columns: {get_feature_columns(fdf)}")
        print(fdf.tail())
