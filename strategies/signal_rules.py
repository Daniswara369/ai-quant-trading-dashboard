"""
Convert ML predictions into actionable trading signals.
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIGNAL_BUY_THRESHOLD, SIGNAL_SELL_THRESHOLD


def generate_signal_from_probability(
    probability_up: float,
    buy_threshold: float = None,
    sell_threshold: float = None,
) -> str:
    """
    Generate a trading signal from model prediction probability.
    
    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    buy_thresh = buy_threshold or SIGNAL_BUY_THRESHOLD
    sell_thresh = sell_threshold or SIGNAL_SELL_THRESHOLD
    
    probability_down = 1.0 - probability_up
    
    if probability_up > buy_thresh:
        return "BUY"
    elif probability_down > sell_thresh:
        return "SELL"
    else:
        return "HOLD"


def generate_signals_batch(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    symbol: str,
    buy_threshold: float = None,
    sell_threshold: float = None,
) -> pd.DataFrame:
    """
    Generate signals for a batch of predictions.
    
    Args:
        df: DataFrame with Close prices and DateTime index.
        probabilities: Array of up-probabilities from model.
        symbol: Asset symbol.
    
    Returns:
        DataFrame with signal details.
    """
    buy_thresh = buy_threshold or SIGNAL_BUY_THRESHOLD
    sell_thresh = sell_threshold or SIGNAL_SELL_THRESHOLD
    
    # Align lengths
    n = min(len(df), len(probabilities))
    df_aligned = df.iloc[-n:].copy()
    probs = probabilities[-n:]
    
    signals = []
    for i in range(len(df_aligned)):
        signal = generate_signal_from_probability(probs[i], buy_thresh, sell_thresh)
        signals.append({
            "timestamp": df_aligned.index[i],
            "symbol": symbol,
            "entry_price": df_aligned["Close"].iloc[i],
            "signal": signal,
            "probability_up": round(float(probs[i]), 4),
            "probability_down": round(1.0 - float(probs[i]), 4),
        })
    
    signals_df = pd.DataFrame(signals)
    return signals_df


def filter_actionable_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out HOLD signals, keeping only BUY and SELL."""
    return signals_df[signals_df["signal"] != "HOLD"].reset_index(drop=True)


def print_signal_summary(signals_df: pd.DataFrame):
    """Print a summary of generated signals."""
    total = len(signals_df)
    buys = len(signals_df[signals_df["signal"] == "BUY"])
    sells = len(signals_df[signals_df["signal"] == "SELL"])
    holds = len(signals_df[signals_df["signal"] == "HOLD"])
    
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║        SIGNAL SUMMARY                ║")
    print(f"╠══════════════════════════════════════╣")
    print(f"║  Total signals : {total:<20d}║")
    print(f"║  BUY signals   : {buys:<20d}║")
    print(f"║  SELL signals  : {sells:<20d}║")
    print(f"║  HOLD signals  : {holds:<20d}║")
    print(f"╚══════════════════════════════════════╝")
