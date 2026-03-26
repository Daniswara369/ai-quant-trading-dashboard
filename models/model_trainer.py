"""
ML Model Trainer — supports XGBoost, Random Forest, LightGBM, and LSTM.
Binary classification: predict next-candle direction (up/down).
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_SAVE_DIR, XGBOOST_PARAMS, RANDOM_FOREST_PARAMS,
    LIGHTGBM_PARAMS, LSTM_PARAMS,
)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: 1 if next-candle close > current close, else 0.
    """
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(subset=["Target"], inplace=True)
    df["Target"] = df["Target"].astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (exclude raw OHLCV and target)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "Target",
               "DateTime", "Market_Regime", "Regime_Label"}
    return [col for col in df.columns if col not in exclude]


def prepare_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Prepare train/test splits with feature scaling.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_columns
    """
    df = create_target(df)
    feature_cols = get_feature_columns(df)
    
    X = df[feature_cols].values
    y = df["Target"].values
    
    # Time-series split (no shuffle to preserve temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_xgboost(X_train, y_train, tune: bool = False):
    """Train XGBoost classifier."""
    from xgboost import XGBClassifier
    
    if tune:
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        grid = GridSearchCV(
            model,
            XGBOOST_PARAMS,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        print(f"  Best XGBoost params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return model


def train_random_forest(X_train, y_train, tune: bool = False):
    """Train Random Forest classifier."""
    from sklearn.ensemble import RandomForestClassifier
    
    if tune:
        model = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(
            model, RANDOM_FOREST_PARAMS, cv=3,
            scoring="f1", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        print(f"  Best RF params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        )
        model.fit(X_train, y_train)
        return model


def train_lightgbm(X_train, y_train, tune: bool = False):
    """Train LightGBM classifier."""
    from lightgbm import LGBMClassifier
    
    if tune:
        model = LGBMClassifier(random_state=42, verbose=-1)
        grid = GridSearchCV(
            model, LIGHTGBM_PARAMS, cv=3,
            scoring="f1", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        print(f"  Best LightGBM params: {grid.best_params_}")
        return grid.best_estimator_
    else:
        model = LGBMClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train)
        return model


def train_lstm(X_train, y_train, X_test=None, y_test=None):
    """Train LSTM neural network."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    seq_len = LSTM_PARAMS["sequence_length"]
    
    # Reshape to sequences
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    
    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)
    
    model = Sequential([
        LSTM(LSTM_PARAMS["units"], return_sequences=True,
             input_shape=(seq_len, X_train.shape[1])),
        Dropout(LSTM_PARAMS["dropout"]),
        LSTM(LSTM_PARAMS["units"] // 2),
        Dropout(LSTM_PARAMS["dropout"]),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
    
    validation_data = None
    if X_test is not None and y_test is not None:
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
        if len(X_test_seq) > 0:
            validation_data = (X_test_seq, y_test_seq)
    
    model.fit(
        X_seq, y_seq,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )
    
    return model


def evaluate_model(model, X_test, y_test, model_type: str = "tree") -> dict:
    """
    Evaluate model and return metrics.
    """
    if model_type == "lstm":
        seq_len = LSTM_PARAMS["sequence_length"]
        Xs = []
        ys = []
        for i in range(seq_len, len(X_test)):
            Xs.append(X_test[i - seq_len:i])
            ys.append(y_test[i])
        if len(Xs) == 0:
            return {"error": "Not enough test data for LSTM evaluation"}
        Xs = np.array(Xs)
        ys = np.array(ys)
        y_proba = model.predict(Xs, verbose=0).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        y_test_eval = ys
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_test_eval = y_test
    
    metrics = {
        "accuracy": float(accuracy_score(y_test_eval, y_pred)),
        "precision": float(precision_score(y_test_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test_eval, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test_eval, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_eval, y_proba)),
    }
    
    print("\n╔══════════════════════════════════════╗")
    print("║       MODEL EVALUATION RESULTS       ║")
    print("╠══════════════════════════════════════╣")
    for k, v in metrics.items():
        print(f"║  {k:<15s}: {v:.4f}             ║")
    print("╚══════════════════════════════════════╝")
    
    return metrics


def save_model(model, scaler, feature_cols, metrics, symbol, model_type, timeframe):
    """Save model, scaler, metadata to disk."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    prefix = f"{symbol}_{model_type}_{timeframe}"
    
    if model_type == "lstm":
        model_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_model.keras")
        model.save(model_path)
    else:
        model_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_model.joblib")
        joblib.dump(model, model_path)
    
    scaler_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    meta = {
        "symbol": symbol,
        "model_type": model_type,
        "timeframe": timeframe,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[SAVED] Model → {model_path}")
    print(f"[SAVED] Scaler → {scaler_path}")
    print(f"[SAVED] Meta   → {meta_path}")
    
    return model_path


def load_model(symbol: str, model_type: str, timeframe: str):
    """Load a saved model, scaler, and metadata."""
    prefix = f"{symbol}_{model_type}_{timeframe}"
    
    if model_type == "lstm":
        import tensorflow as tf
        model_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_model.keras")
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_model.joblib")
        model = joblib.load(model_path)
    
    scaler_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    meta_path = os.path.join(MODEL_SAVE_DIR, f"{prefix}_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    return model, scaler, meta


def train_pipeline(
    df: pd.DataFrame,
    symbol: str,
    model_type: str = "xgboost",
    timeframe: str = "1h",
    tune: bool = False,
) -> dict:
    """
    Full training pipeline: prepare → train → evaluate → save.
    
    Returns:
        Dictionary with model, metrics, and paths.
    """
    print(f"\n{'='*50}")
    print(f"  Training {model_type.upper()} for {symbol} ({timeframe})")
    print(f"{'='*50}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")
    
    # Train model
    print(f"\n  Training {model_type}...")
    if model_type == "xgboost":
        model = train_xgboost(X_train, y_train, tune=tune)
        eval_type = "tree"
    elif model_type == "random_forest":
        model = train_random_forest(X_train, y_train, tune=tune)
        eval_type = "tree"
    elif model_type == "lightgbm":
        model = train_lightgbm(X_train, y_train, tune=tune)
        eval_type = "tree"
    elif model_type == "lstm":
        model = train_lstm(X_train, y_train, X_test, y_test)
        eval_type = "lstm"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, model_type=eval_type)
    
    # Save
    model_path = save_model(model, scaler, feature_cols, metrics, symbol, model_type, timeframe)
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "model_path": model_path,
    }
