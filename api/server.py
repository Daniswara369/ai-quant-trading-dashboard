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
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime

import config as app_config
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

from agent_consensus import (
    AgentVote,
    SignalLogStore,
    build_committee_snapshot,
    committee_agreement_metrics,
)

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
        "http://localhost:6001",
        "http://127.0.0.1:6001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_ASSETS = ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT", "TSLA", "EURUSD", "GBPUSD", "USDJPY"]
RESULTS_DIR = os.path.join(PROJECT_ROOT, "backtesting", "results")


def _default_committee() -> List[str]:
    try:
        parsed = json.loads(app_config.AGENT_COMMITTEE_JSON)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return ["Analyst", "Sentiment Strategist", "Risk Auditor"]


class AgentVotePayload(BaseModel):
    agent_name: str = Field(..., min_length=1)
    direction: int = Field(..., ge=-1, le=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    entry_price: float = Field(..., gt=0)
    symbol: str = Field(..., min_length=1)
    timeframe: str = "1h"
    decision_id: Optional[str] = None

    @field_validator("agent_name", "symbol", mode="before")
    @classmethod
    def strip_names(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("expected string")
        s = v.strip()
        if not s:
            raise ValueError("must not be empty")
        return s


class ResolveMarkoutPayload(BaseModel):
    decision_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)

    @field_validator("decision_id", "agent_name", mode="before")
    @classmethod
    def strip_ids(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("expected string")
        s = v.strip()
        if not s:
            raise ValueError("must not be empty")
        return s

    price_5m: Optional[float] = Field(None, gt=0)
    price_15m: Optional[float] = Field(None, gt=0)


class AgreementPayload(BaseModel):
    """Current-bar votes for agreement metrics only (not persisted)."""

    votes: List[dict]  # [{"direction": 1, "confidence": 0.7}, ...]

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

def _symbol_news_query(symbol: str, market: str) -> str:
    """Build market-aware search query so headlines stay instrument-specific."""
    s = (symbol or "").upper().strip()

    # Common aliases for higher precision on free news endpoints
    crypto_alias = {
        "BTCUSDT": "Bitcoin OR BTC",
        "ETHUSDT": "Ethereum OR ETH",
    }
    stock_alias = {
        "AAPL": "Apple OR AAPL",
        "MSFT": "Microsoft OR MSFT",
        "TSLA": "Tesla OR TSLA",
    }
    forex_alias = {
        "EURUSD": "EURUSD OR euro dollar OR ECB OR Fed",
        "GBPUSD": "GBPUSD OR pound dollar OR BOE OR Fed",
        "USDJPY": "USDJPY OR dollar yen OR BOJ OR Fed",
        "AUDUSD": "AUDUSD OR aussie dollar OR RBA OR Fed",
        "USDCAD": "USDCAD OR dollar loonie OR BOC OR Fed OR oil",
        "USDCHF": "USDCHF OR dollar swiss franc OR SNB OR Fed",
        "NZDUSD": "NZDUSD OR kiwi dollar OR RBNZ OR Fed",
    }

    if market == "crypto":
        base = s.replace("USDT", "").replace("USD", "")
        return crypto_alias.get(s, f"{base} OR {s}")
    if market == "stock":
        return stock_alias.get(s, f"{s} stock OR {s} earnings")
    if market == "forex":
        return forex_alias.get(s, f"{s} OR central bank OR rates")
    return s


def fetch_news_for_symbol(symbol: str) -> list[str]:
    """
    Fetch instrument-related headlines using free APIs.
    Priority:
    1) CryptoCompare for crypto symbols
    2) GDELT DOC query tuned to the chosen instrument (all markets)
    """
    market = detect_market_type(symbol)
    s = (symbol or "").upper().strip()
    results: List[str] = []

    # 1) Fast crypto-specific feed
    if market == "crypto":
        base_asset = s.replace("USDT", "").replace("USD", "")
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={base_asset}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("Data"):
                    for item in data["Data"][:6]:
                        title = (item.get("title") or "").strip()
                        if title:
                            results.append(title)
        except Exception as e:
            logger.warning(f"CryptoCompare news fetch failed for {symbol}: {e}")

    # 2) Instrument-aware global news (free, no key)
    query = _symbol_news_query(s, market)
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 8,
            "sort": "HybridRel",
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            arts = data.get("articles") or []
            for a in arts:
                title = (a.get("title") or "").strip()
                if title:
                    results.append(title)
    except Exception as e:
        logger.warning(f"GDELT symbol news fetch failed for {symbol}: {e}")

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for t in results:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    if deduped:
        return deduped[:8]

    return [
        f"{symbol} news flow is currently thin; awaiting fresh instrument-specific catalysts.",
        f"{symbol} trades in a macro-sensitive environment with mixed short-term headlines.",
    ]


def fetch_global_macro_headlines() -> list[str]:
    """
    Fetches top global-macro headlines using a free, public API.
    Uses GDELT 2 DOC endpoint (no API key required) and focuses on 2026-relevant macro triggers.

    Target themes: Strait of Hormuz, Fed policy, OPEC/energy supply.
    """
    query = "Hormuz OR Fed OR OPEC"
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 3,
            "sort": "HybridRel",
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            arts = data.get("articles") or []
            titles = []
            for a in arts[:3]:
                t = (a.get("title") or "").strip()
                if t:
                    titles.append(t)
            if titles:
                return titles
    except Exception as e:
        logger.warning(f"Global macro news fetch failed: {e}")

    return [
        "Fed officials reiterate data-dependent stance as inflation risks persist.",
        "OPEC+ signals continued supply discipline amid volatile energy markets.",
        "Strait of Hormuz remains a key energy chokepoint; risk premium stays elevated.",
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
        await manager.broadcast(json.dumps({"type": "info", "message": f"No data available for {symbol}"}))
        raise HTTPException(status_code=404, detail="No data available")
        
    df = engineer_features(df)
    current_price = float(df["Close"].iloc[-1])
    
    # ── Auto-resolve old markouts with current price ──
    try:
        store = SignalLogStore(app_config.AGENT_SIGNAL_LOG_PATH)
        all_rows = store.load_all()
        for row in all_rows:
            if row.resolved_at is None and row.entry_price > 0:
                # Resolve with current price for both 5m and 15m
                if row.price_5m is None:
                    store.update_row_prices(row.decision_id, row.agent_name, price_5m=current_price)
                if row.price_15m is None:
                    store.update_row_prices(row.decision_id, row.agent_name, price_15m=current_price)
        logger.info(f"Auto-resolved outstanding markouts with price {current_price}")
    except Exception as e:
        logger.warning(f"Auto-resolve markouts failed (non-fatal): {e}")
    
    # Needs to match AgentContext expectations
    context = AgentContext(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_df=df,
        indicators={},
        timestamp=datetime.now(),
        metadata={
            "news_headlines": fetch_news_for_symbol(symbol),
            "global_headlines": fetch_global_macro_headlines(),
        }
    )
    
    # Broadcast starting message
    await manager.broadcast(json.dumps({"type": "info", "message": f"Starting agent analysis for {symbol}"}))
    
    try:
        consensus = supervisor.analyze_and_debate(context)
        
        # Sanitize the consensus dict for JSON serialization
        consensus_data = consensus.dict()
        consensus_data = _sanitize_for_json(consensus_data)
        
        last_agent_status = consensus_data
        
        # Broadcast result
        await manager.broadcast(json.dumps({
            "type": "consensus",
            "data": last_agent_status
        }))
        
        return last_agent_status
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Agent analysis failed: {error_msg}\n{traceback.format_exc()}")
        
        # Broadcast error so the frontend stops spinning
        await manager.broadcast(json.dumps({
            "type": "consensus",
            "data": {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {error_msg[:200]}. Using fallback.",
                "agent_outputs": [],
                "regime": {"regime": "Default", "confidence": 0.5, "reasoning": "Fallback due to error."}
            }
        }))
        
        return {"signal": "HOLD", "confidence": 0.0, "reasoning": f"Error: {error_msg[:200]}"}


def _sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # Catch any remaining numpy scalar
        return obj.item()
    return obj

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


@app.get("/api/agent-committee/snapshot")
def get_agent_committee_snapshot(
    alpha: float = Query(0.15, ge=0.01, le=1.0),
    temperature: float = Query(4.0, gt=0, le=50.0),
    min_weight: float = Query(0.15, gt=0, lt=1),
    max_weight: float = Query(0.55, gt=0, le=1),
):
    """
    Deterministic EMA performance + capped weights from `AGENT_SIGNAL_LOG` JSONL.
    Optional env `AGENT_COMMITTEE_JSON` lists agent names for the committee.
    """
    committee = _default_committee()
    store = SignalLogStore(app_config.AGENT_SIGNAL_LOG_PATH)
    try:
        snap = build_committee_snapshot(
            store,
            committee,
            alpha=alpha,
            temperature=temperature,
            min_weight=min_weight,
            max_weight=max_weight,
        )
    except ValueError as e:
        msg = str(e)
        # Corrupt on-disk log is a server/data integrity issue, not a client mistake
        status = 500 if "Invalid agent signal log line" in msg else 400
        raise HTTPException(status_code=status, detail=msg)
    snap["committee"] = committee
    snap["log_path"] = app_config.AGENT_SIGNAL_LOG_PATH
    return snap


@app.post("/api/agent-committee/vote")
def post_agent_vote(payload: AgentVotePayload):
    """Append one validated agent vote row to the JSONL log."""
    store = SignalLogStore(app_config.AGENT_SIGNAL_LOG_PATH)
    kw = dict(
        agent_name=payload.agent_name,
        direction=payload.direction,
        confidence=payload.confidence,
        entry_price=payload.entry_price,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
    )
    try:
        if payload.decision_id is not None and str(payload.decision_id).strip():
            did = str(payload.decision_id).strip()
            vote = AgentVote(**kw, decision_id=did)
        else:
            vote = AgentVote(**kw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rec = store.append_vote(vote)
    return {"status": "ok", "record": rec.to_dict()}


@app.post("/api/agent-committee/resolve-markout")
def post_resolve_markout(payload: ResolveMarkoutPayload):
    """Attach 5m/15m prices to an existing row; recomputes markouts."""
    if payload.price_5m is None and payload.price_15m is None:
        raise HTTPException(status_code=400, detail="Provide price_5m and/or price_15m")
    store = SignalLogStore(app_config.AGENT_SIGNAL_LOG_PATH)
    n = store.update_row_prices(
        payload.decision_id,
        payload.agent_name,
        price_5m=payload.price_5m,
        price_15m=payload.price_15m,
    )
    if n == 0:
        raise HTTPException(status_code=404, detail="No matching decision_id + agent_name")
    return {"status": "ok", "updated": n}


@app.post("/api/agent-committee/agreement")
def post_committee_agreement(payload: AgreementPayload):
    """Compute confidence-weighted agreement for a set of live votes (no DB write)."""
    votes = []
    for v in payload.votes:
        d = int(v.get("direction", 0))
        c = float(v.get("confidence", 0.0))
        votes.append((d, c))
    return committee_agreement_metrics(votes)


@app.get("/")
def root():
    return {
        "name": "AI Quant Trading System API",
        "version": "2.0.0",
        "docs": "/docs",
    }
