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

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import pandas as pd
import numpy as np

import config as app_config
from config import (
    ALL_ASSETS, SUPPORTED_TIMEFRAMES, CRYPTO_ASSETS,
    STOCK_ASSETS, FOREX_ASSETS, MODEL_SAVE_DIR, LSTM_PARAMS,
)
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
    allow_origins=["*"],
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
    return ["TechnicalAnalyst", "SentimentStrategist", "RiskAuditor"]


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
