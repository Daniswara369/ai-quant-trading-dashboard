"""
Real-time market monitor — periodically fetches data, updates indicators,
runs ML inference, and generates trading signals.
"""
import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_pipeline import fetch_data
from feature_engineering import engineer_features
from models.model_trainer import load_model
from strategies.signal_rules import generate_signal_from_probability
from config import REALTIME_REFRESH_SECONDS, LSTM_PARAMS
import numpy as np


class RealtimeMonitor:
    """
    Real-time market monitor that:
    1. Fetches latest candle data
    2. Recomputes indicators
    3. Runs ML inference
    4. Emits signals
    """
    
    def __init__(
        self,
        symbols: list = None,
        market_types: dict = None,
        timeframe: str = "1h",
        model_type: str = "xgboost",
        refresh_seconds: int = None,
    ):
        from config import CRYPTO_ASSETS, STOCK_ASSETS, FOREX_ASSETS
        
        self.symbols = symbols or CRYPTO_ASSETS[:1]  # Default: just BTCUSDT
        self.market_types = market_types or {}
        self.timeframe = timeframe
        self.model_type = model_type
        self.refresh_seconds = refresh_seconds or REALTIME_REFRESH_SECONDS
        
        # Pre-load models
        self.models = {}
        self.scalers = {}
        self.metas = {}
        
        for symbol in self.symbols:
            try:
                model, scaler, meta = load_model(symbol, model_type, timeframe)
                self.models[symbol] = model
                self.scalers[symbol] = scaler
                self.metas[symbol] = meta
                print(f"  [LOADED] Model for {symbol}")
            except FileNotFoundError:
                print(f"  [SKIP] No model for {symbol} — train it first")
        
        self.signal_log = []
    
    def _predict_single(self, symbol: str, df):
        """Run prediction for a single asset."""
        if symbol not in self.models:
            return None, None
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        feature_cols = self.metas[symbol]["feature_columns"]
        
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        if self.model_type == "lstm":
            seq_len = LSTM_PARAMS["sequence_length"]
            if len(X_scaled) < seq_len:
                return None, None
            X_seq = np.array([X_scaled[-seq_len:]])
            prob = model.predict(X_seq, verbose=0).flatten()[0]
        else:
            prob = model.predict_proba(X_scaled[-1:])[:, 1][0]
        
        signal = generate_signal_from_probability(prob)
        return signal, prob
    
    def scan_once(self):
        """Run one scan cycle across all symbols."""
        print(f"\n{'─'*55}")
        print(f"  REAL-TIME SCAN @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─'*55}")
        
        for symbol in self.symbols:
            market = self.market_types.get(symbol)
            
            # Fetch latest data
            df = fetch_data(symbol, market, self.timeframe, force_refresh=True)
            if df.empty:
                print(f"  {symbol}: no data")
                continue
            
            # Engineer features
            df = engineer_features(df, drop_na=True)
            if df.empty:
                print(f"  {symbol}: insufficient data after feature engineering")
                continue
            
            # Predict
            signal, prob = self._predict_single(symbol, df)
            if signal is None:
                print(f"  {symbol}: no model loaded")
                continue
            
            current_price = df["Close"].iloc[-1]
            
            # Emoji markers
            if signal == "BUY":
                icon = "🟢"
            elif signal == "SELL":
                icon = "🔴"
            else:
                icon = "⚪"
            
            print(f"  {icon} {symbol:<10s} | Price: {current_price:>12.4f} | {signal:<4s} | Prob Up: {prob:.3f}")
            
            # Log
            self.signal_log.append({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "price": current_price,
                "signal": signal,
                "probability_up": round(float(prob), 4),
            })
    
    def run_loop(self):
        """Run continuous monitoring loop."""
        print(f"\n{'═'*55}")
        print(f"  REAL-TIME SIGNAL ENGINE STARTED")
        print(f"  Monitoring: {', '.join(self.symbols)}")
        print(f"  Timeframe: {self.timeframe} | Model: {self.model_type}")
        print(f"  Refresh: every {self.refresh_seconds}s")
        print(f"  Press Ctrl+C to stop")
        print(f"{'═'*55}")
        
        try:
            while True:
                self.scan_once()
                print(f"\n  Next scan in {self.refresh_seconds}s...")
                time.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            print(f"\n\n  ■ Monitoring stopped. Total signals logged: {len(self.signal_log)}")
