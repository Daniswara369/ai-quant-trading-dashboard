"""
Feature importance analysis for trained tree-based models.
Uses built-in importance and SHAP-style analysis.
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_feature_importance(model, feature_columns: list, model_type: str = "xgboost") -> pd.DataFrame:
    """
    Extract feature importance from a trained tree model.
    
    Returns:
        DataFrame with Feature and Importance columns, sorted descending.
    """
    if model_type == "lstm":
        print("[INFO] Feature importance not directly available for LSTM models.")
        return pd.DataFrame({"Feature": feature_columns, "Importance": [1.0 / len(feature_columns)] * len(feature_columns)})
    
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("[WARNING] Model doesn't support feature_importances_")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances,
    })
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    
    return importance_df


def print_top_features(importance_df: pd.DataFrame, top_n: int = 20):
    """Pretty-print the top N important features."""
    if importance_df.empty:
        return
    
    print(f"\n{'='*45}")
    print(f"  TOP {top_n} FEATURE IMPORTANCE")
    print(f"{'='*45}")
    
    for i, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"  {row['Feature']:<25s} {row['Importance']:.4f} {bar}")
    
    print(f"{'='*45}")
