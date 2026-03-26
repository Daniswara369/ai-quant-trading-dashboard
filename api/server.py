"""
FastAPI backend for the AI Quant Trading System.
Wraps existing trading engine modules as REST endpoints.

Run:
    uvicorn api.server:app --reload --port 8000
"""
import sys
import os
import json
import traceback
import requests
import logging

logger = logging.getLogger(__name__)

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime

from config import (
    ALL_ASSETS, SUPPORTED_TIMEFRAMES, CRYPTO_ASSETS,
    STOCK_ASSETS, FOREX_ASSETS, MODEL_SAVE_DIR, LSTM_PARAMS, STALE_SIGNAL_THRESHOLD
)
from agents.supervisor_agent import SupervisorAgent
from core.schemas import AgentContext, TradeExecution
from data.data_pipeline import fetch_data, detect_market_type
from feature_engineering import engineer_features
from models.model_trainer import load_model
from strategies.signal_rules import generate_signal_from_probability
from backtesting.backtest_engine import BacktestEngine

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI Quant Trading System API",
    description="REST API for ML-powered trading signal generation and backtesting.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_ASSETS = ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT", "TSLA", "EURUSD", "GBPUSD", "USDJPY"]
RESULTS_DIR = os.path.join(PROJECT_ROOT, "backtesting", "results")

# ──────────────────────────────────────────────
# Agent State & WebSocket Manager
# ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()
supervisor = SupervisorAgent()
last_agent_status = None

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_signals_for_asset(symbol: str, timeframe: str = "1h", model_type: str = "xgboost"):
    """Load model, fetch data, engineer features, predict, return signals + df."""
    market = detect_market_type(symbol)
    model, scaler, meta = load_model(symbol, model_type, timeframe)
    feature_cols = meta["feature_columns"]

    df = fetch_data(symbol, market, timeframe)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = engineer_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    signals = [generate_signal_from_probability(p) for p in probs]

    return df, probs, signals, meta, market


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/api/assets")
def get_assets():
    """Return list of supported instruments grouped by market."""
    return {
        "assets": DASHBOARD_ASSETS,
        "grouped": {
            "crypto": CRYPTO_ASSETS,
            "stock": STOCK_ASSETS,
            "forex": ["EURUSD", "GBPUSD", "USDJPY"],
        },
        "timeframes": SUPPORTED_TIMEFRAMES,
    }


@app.get("/api/market-data")
def get_market_data(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
    limit: int = Query(200, ge=10, le=2000),
):
    """Return OHLCV + indicator data for chart rendering."""
    market = detect_market_type(symbol)
    df = fetch_data(symbol, market, timeframe)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = engineer_features(df)
    df = df.tail(limit)

    # Convert to JSON-safe format
    records = []
    for idx, row in df.iterrows():
        rec = {"timestamp": str(idx)}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = round(float(val), 6)
            else:
                rec[col] = val
        records.append(rec)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "market": market,
        "count": len(records),
        "data": records,
    }

def fetch_news_for_symbol(symbol: str) -> list[str]:
    """Fetches recent news headlines for a given symbol using a free, public API."""
    base_asset = symbol.replace("USDT", "").replace("USD", "").upper()
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={base_asset}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("Data"):
                return [item["title"] for item in data["Data"][:5]]
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
    
    return [
        f"Market awaits fresh macroeconomic catalysts for {symbol}.", 
        f"Trading volume for {symbol} remains steady amidst uncertainty."
    ]

@app.websocket("/ws/agent-thoughts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/agent-signal")
async def run_agent_signal(
    background_tasks: BackgroundTasks,
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h")
):
    """Run full agent pipeline (debate loop) -> returns weighted consensus."""
    global last_agent_status
    
    market = detect_market_type(symbol)
    df = fetch_data(symbol, market, timeframe)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")
        
    df = engineer_features(df)
    
    # Needs to match AgentContext expectations
    context = AgentContext(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_df=df,
        indicators={},
        timestamp=datetime.now(),
        metadata={
            "news_headlines": fetch_news_for_symbol(symbol)
        }
    )
    
    # Broadcast starting message
    await manager.broadcast(json.dumps({"type": "info", "message": f"Starting agent analysis for {symbol}"}))
    
    consensus = supervisor.analyze_and_debate(context)
    last_agent_status = consensus.dict()
    
    # Broadcast result
    await manager.broadcast(json.dumps({
        "type": "consensus",
        "data": last_agent_status
    }))
    
    return last_agent_status

@app.get("/api/agent-status")
def get_agent_status():
    """Return the last pipeline result."""
    if not last_agent_status:
         return {"status": "no data"}
    return last_agent_status

@app.post("/api/mock-trade")
def execute_mock_trade(
    symbol: str = Query("BTCUSDT"),
    trigger_price: float = Query(0.0),
    signal: str = Query("HOLD"),
    confidence: float = Query(0.0),
    timeframe: str = Query("1h")
):
    """Execute mock trade based on Gemini function call or frontend command, with stale signal guard."""
    market = detect_market_type(symbol)
    df = fetch_data(symbol, market, timeframe)
    if df.empty:
        raise HTTPException(status_code=400, detail="No price data available")
        
    current_price = float(df["Close"].iloc[-1])
    
    if trigger_price > 0:
        deviation = abs(current_price - trigger_price) / trigger_price
        if deviation > STALE_SIGNAL_THRESHOLD:
            raise HTTPException(status_code=400, detail=f"Stale signal: price deviated by {deviation:.4%}, threshold is {STALE_SIGNAL_THRESHOLD:.4%}")
        
    execution = TradeExecution(
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        trigger_price=trigger_price or current_price,
        position_size=1.0, # Dummy size
        stop_loss=current_price * 0.98 if signal == "BUY" else current_price * 1.02,
        take_profit=current_price * 1.04 if signal == "BUY" else current_price * 0.96,
        reasoning="Mock execution",
        timestamp=datetime.now()
    )
    
    return {"status": "success", "execution": execution.dict()}

@app.get("/api/signals")
def get_signals(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
    model_type: str = Query("xgboost"),
    limit: int = Query(100),
):
    """Return ML trading signals for a given asset."""
    try:
        df, probs, signals, meta, market = _get_signals_for_asset(symbol, timeframe, model_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No trained model for {symbol}/{model_type}/{timeframe}")

    n = min(limit, len(df))
    df_tail = df.tail(n)
    probs_tail = probs[-n:]
    sigs_tail = signals[-n:]

    records = []
    for i, (idx, row) in enumerate(df_tail.iterrows()):
        records.append({
            "timestamp": str(idx),
            "price": round(float(row["Close"]), 6),
            "signal": sigs_tail[i],
            "probability_up": round(float(probs_tail[i]), 4),
            "probability_down": round(1.0 - float(probs_tail[i]), 4),
        })

    # Latest summary
    latest = records[-1] if records else {}

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "model": model_type,
        "latest": latest,
        "signals": records,
        "training_metrics": meta.get("metrics", {}),
    }


@app.get("/api/prediction")
def get_prediction(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
    model_type: str = Query("xgboost"),
):
    """Return the latest ML prediction for an asset."""
    try:
        df, probs, signals, meta, market = _get_signals_for_asset(symbol, timeframe, model_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No trained model for {symbol}")

    latest_prob = float(probs[-1])
    latest_signal = signals[-1]
    latest_price = float(df["Close"].iloc[-1])

    return {
        "symbol": symbol,
        "current_price": round(latest_price, 6),
        "signal": latest_signal,
        "probability_up": round(latest_prob, 4),
        "probability_down": round(1.0 - latest_prob, 4),
        "model": model_type,
        "training_metrics": meta.get("metrics", {}),
    }


@app.get("/api/all-signals")
def get_all_signals(
    timeframe: str = Query("1h"),
    model_type: str = Query("xgboost"),
):
    """Return latest signal for ALL dashboard assets (overview table)."""
    results = []
    for symbol in DASHBOARD_ASSETS:
        try:
            df, probs, signals, meta, market = _get_signals_for_asset(symbol, timeframe, model_type)
            latest_price = round(float(df["Close"].iloc[-1]), 6)
            latest_prob = round(float(probs[-1]), 4)
            latest_signal = signals[-1]

            # Load backtest metrics if available
            bt_path = os.path.join(RESULTS_DIR, f"{symbol}_{timeframe}_backtest.json")
            bt_metrics = {}
            if os.path.exists(bt_path):
                with open(bt_path) as f:
                    bt_data = json.load(f)
                    bt_metrics = bt_data.get("metrics", {})

            results.append({
                "symbol": symbol,
                "market": market,
                "price": latest_price,
                "signal": latest_signal,
                "probability_up": latest_prob,
                "sharpe_ratio": bt_metrics.get("sharpe_ratio", 0),
                "total_return_pct": bt_metrics.get("total_return_pct", 0),
                "win_rate_pct": bt_metrics.get("win_rate_pct", 0),
                "max_drawdown_pct": bt_metrics.get("max_drawdown_pct", 0),
            })
        except Exception:
            results.append({
                "symbol": symbol,
                "market": detect_market_type(symbol),
                "price": 0,
                "signal": "N/A",
                "probability_up": 0,
                "sharpe_ratio": 0,
                "total_return_pct": 0,
                "win_rate_pct": 0,
                "max_drawdown_pct": 0,
            })

    return {"assets": results}


@app.get("/api/backtest-results")
def get_backtest_results(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
):
    """Return cached backtest results."""
    bt_path = os.path.join(RESULTS_DIR, f"{symbol}_{timeframe}_backtest.json")
    if not os.path.exists(bt_path):
        raise HTTPException(status_code=404, detail=f"No backtest results for {symbol}. Run train_all_models.py first.")

    with open(bt_path) as f:
        data = json.load(f)

    return data


@app.get("/api/equity-curve")
def get_equity_curve(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
):
    """Return equity curve data from cached backtest."""
    bt_path = os.path.join(RESULTS_DIR, f"{symbol}_{timeframe}_backtest.json")
    if not os.path.exists(bt_path):
        raise HTTPException(status_code=404, detail=f"No backtest results for {symbol}")

    with open(bt_path) as f:
        data = json.load(f)

    return {
        "symbol": symbol,
        "equity_curve": data.get("equity_curve", []),
    }


@app.post("/api/refresh-data")
def refresh_data(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
):
    """Force re-fetch market data from source."""
    market = detect_market_type(symbol)
    df = fetch_data(symbol, market, timeframe, force_refresh=True)
    if df.empty:
        raise HTTPException(status_code=500, detail=f"Failed to refresh data for {symbol}")

    return {
        "status": "ok",
        "symbol": symbol,
        "rows": len(df),
        "latest_timestamp": str(df.index[-1]),
        "latest_close": round(float(df["Close"].iloc[-1]), 6),
    }


@app.post("/api/refresh-signal")
def refresh_signal(
    symbol: str = Query("BTCUSDT"),
    timeframe: str = Query("1h"),
    model_type: str = Query("xgboost"),
):
    """Re-fetch data + re-run ML inference for a fresh signal."""
    # Force refresh data first
    market = detect_market_type(symbol)
    df_raw = fetch_data(symbol, market, timeframe, force_refresh=True)
    if df_raw.empty:
        raise HTTPException(status_code=500, detail=f"Failed to refresh data for {symbol}")

    try:
        df, probs, signals, meta, market = _get_signals_for_asset(symbol, timeframe, model_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No trained model for {symbol}")

    latest_prob = float(probs[-1])
    latest_signal = signals[-1]
    latest_price = float(df["Close"].iloc[-1])

    return {
        "status": "ok",
        "symbol": symbol,
        "current_price": round(latest_price, 6),
        "signal": latest_signal,
        "probability_up": round(latest_prob, 4),
        "probability_down": round(1.0 - latest_prob, 4),
        "latest_timestamp": str(df.index[-1]),
    }


@app.get("/")
def root():
    return {
        "name": "AI Quant Trading System API",
        "version": "2.0.0",
        "docs": "/docs",
    }
