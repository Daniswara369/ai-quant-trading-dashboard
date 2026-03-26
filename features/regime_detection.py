"""
Market regime detection: Bull / Bear / Sideways.
Uses rolling returns and volatility clustering.
"""
import pandas as pd
import numpy as np


def detect_regime(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    Detect market regime based on trend direction and volatility.
    
    Regimes:
        1 = Bull  (positive trend, low-med vol)
        -1 = Bear  (negative trend, any vol)
        0 = Sideways (no clear trend)
    """
    df = df.copy()
    
    # Rolling return (trend direction)
    df["_rolling_return"] = df["Close"].pct_change(window)
    
    # Rolling volatility
    df["_rolling_vol"] = df["Close"].pct_change().rolling(window).std()
    vol_median = df["_rolling_vol"].median()
    
    # Thresholds
    trend_threshold = 0.02  # 2% return over window
    
    conditions = [
        (df["_rolling_return"] > trend_threshold),                              # Bull
        (df["_rolling_return"] < -trend_threshold),                             # Bear
    ]
    choices = [1, -1]
    
    df["Market_Regime"] = np.select(conditions, choices, default=0)
    
    # Labels
    regime_map = {1: "Bull", -1: "Bear", 0: "Sideways"}
    df["Regime_Label"] = df["Market_Regime"].map(regime_map)
    
    # Cleanup temp columns
    df.drop(columns=["_rolling_return", "_rolling_vol"], inplace=True)
    
    return df
