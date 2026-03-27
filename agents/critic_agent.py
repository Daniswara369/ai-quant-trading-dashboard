import logging
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    Institutional-Grade Risk Auditor / 'The Bear'.
    Generates structured Counter-Theses, calculates Risk-to-Reward ratios,
    Expected Value ($EV$), and position sizing based on agent reliability weights.
    Aware of its own Reliability Weight from Jackson's PerformanceTracker.
    """

    def __init__(self):
        self.name = "Risk Auditor"

    # ─────────────────────────────────────────
    # Counter-Thesis Generation
    # ─────────────────────────────────────────

    def _generate_counter_thesis(self, df: pd.DataFrame) -> List[str]:
        """
        Generates structured reasons why the current trade setup could fail.
        Goes beyond simple indicator checks to institutional-grade risk awareness.
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        counter_points = []

        # 1. RSI Exhaustion / Divergence
        if "RSI" in latest:
            rsi = latest["RSI"]
            if rsi > 75:
                counter_points.append(
                    f"EXHAUSTION RISK: RSI at {rsi:.1f} — extreme overbought territory. "
                    f"Historical mean reversion probability is elevated."
                )
            elif rsi < 25:
                counter_points.append(
                    f"KNIFE-CATCHING RISK: RSI at {rsi:.1f} — extreme oversold. "
                    f"Could indicate capitulation selling with further downside."
                )
            # RSI hidden divergence: price makes new high but RSI doesn't
            if rsi < 65 and latest["Close"] > df["Close"].rolling(10).max().iloc[-2]:
                counter_points.append(
                    f"HIDDEN DIVERGENCE: Price making new highs but RSI ({rsi:.1f}) failing to confirm. "
                    f"Momentum is weakening beneath the surface."
                )

        # 2. Bull/Bear Trap Detection (structural)
        if "BB_Upper" in latest and "BB_Lower" in latest:
            if latest["High"] > latest["BB_Upper"] and latest["Close"] < prev["Close"]:
                counter_points.append(
                    "BULL TRAP: Price pierced upper Bollinger Band but closed red. "
                    "Institutional players may have used the breakout to distribute."
                )
            elif latest["Low"] < latest["BB_Lower"] and latest["Close"] > prev["Close"]:
                counter_points.append(
                    "BEAR TRAP: Price pierced lower Bollinger Band but closed green. "
                    "Stop-hunt below support followed by aggressive buying."
                )

        # 3. Volume Divergence (conviction check)
        if "Volume_Ratio" in latest:
            vol_ratio = latest["Volume_Ratio"]
            if latest["Close"] > df["Close"].rolling(10).max().iloc[-2] and vol_ratio < 0.8:
                counter_points.append(
                    f"WEAK CONVICTION: Price at 10-bar high but volume only {vol_ratio:.1f}x average. "
                    f"Breakout lacks institutional participation."
                )
            elif latest["Close"] < df["Close"].rolling(10).min().iloc[-2] and vol_ratio < 0.8:
                counter_points.append(
                    f"THIN SELLING: Price at 10-bar low but volume only {vol_ratio:.1f}x average. "
                    f"Breakdown may lack follow-through."
                )

        # 4. Volatility Regime Shift
        if "Rolling_Volatility" in latest and "ATR" in latest and "ATR" in df.columns:
            atr_avg = df["ATR"].rolling(14).mean().iloc[-1]
            if latest["ATR"] > atr_avg * 1.5:
                counter_points.append(
                    f"VOLATILITY SPIKE: ATR ({latest['ATR']:.4f}) is 1.5x the 14-period mean ({atr_avg:.4f}). "
                    f"Increased risk of whipsaw and stop-hunting."
                )

        # 5. Momentum Exhaustion
        if "Momentum_5" in latest and "Momentum_20" in latest:
            m5 = latest["Momentum_5"]
            m20 = latest["Momentum_20"]
            if m5 > 0 and m20 < 0:
                counter_points.append(
                    "MOMENTUM DIVERGENCE: Short-term momentum positive but long-term negative. "
                    "Current bounce may be a dead-cat bounce within a larger downtrend."
                )
            elif m5 < 0 and m20 > 0:
                counter_points.append(
                    "PULLBACK IN UPTREND: Short-term momentum negative within a positive long-term trend. "
                    "Could be a buying opportunity — or the start of a reversal."
                )

        if not counter_points:
            counter_points.append("No significant structural risks identified in the current setup.")

        return counter_points[:4]  # Top 4 most critical risks

    # ─────────────────────────────────────────
    # Risk-to-Reward Calculation
    # ─────────────────────────────────────────

    def _calculate_risk_reward(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates Risk-to-Reward ratio using ATR-based stop loss and take profit.
        Stop Loss = 1.5 × ATR below entry
        Take Profit = 2.0 × ATR above entry (for longs)
        """
        latest = df.iloc[-1]
        atr = latest.get("ATR", 0)
        close = latest.get("Close", 0)

        if atr == 0 or close == 0:
            return {"rr_ratio": 0, "stop_loss": 0, "take_profit": 0, "risk_pct": 0}

        stop_distance = 1.5 * atr
        reward_distance = 2.0 * atr

        rr_ratio = reward_distance / stop_distance if stop_distance > 0 else 0

        return {
            "rr_ratio": round(rr_ratio, 2),
            "stop_loss": round(close - stop_distance, 6),
            "take_profit": round(close + reward_distance, 6),
            "stop_distance": round(stop_distance, 6),
            "reward_distance": round(reward_distance, 6),
            "risk_pct": round((stop_distance / close) * 100, 2),
        }

    # ─────────────────────────────────────────
    # Expected Value Calculation
    # ─────────────────────────────────────────

    def _calculate_expected_value(self, rr: Dict[str, Any], analyst_win_rate: float) -> Dict[str, Any]:
        """
        EV = (Win% × Reward) - (Loss% × Risk)
        Uses the Analyst's EMA win rate from Jackson's PerformanceTracker.
        """
        win_rate = max(analyst_win_rate, 0.01)  # Floor at 1%
        loss_rate = 1.0 - win_rate
        rr_ratio = rr.get("rr_ratio", 0)

        ev = (win_rate * rr_ratio) - (loss_rate * 1.0)

        return {
            "ev": round(ev, 4),
            "ev_positive": ev > 0,
            "win_rate_used": round(win_rate, 3),
            "interpretation": (
                f"EV = ({win_rate:.1%} × {rr_ratio:.2f}R) - ({loss_rate:.1%} × 1R) = {ev:+.4f}R. "
                + ("Trade has positive expected value." if ev > 0 else "⚠️ Trade has NEGATIVE expected value — avoid or reduce size.")
            ),
        }

    # ─────────────────────────────────────────
    # Position Sizing (Half-Kelly)
    # ─────────────────────────────────────────

    def _suggest_position_size(self, ev_data: Dict, analyst_weight: float) -> Dict[str, Any]:
        """
        Half-Kelly position sizing adjusted by the Analyst's reliability weight.
        Kelly% = (Win% × R - Loss%) / R
        Position = Half-Kelly × Analyst_Weight (more trust = larger size)
        """
        win_rate = ev_data.get("win_rate_used", 0.5)
        rr_ratio = max(ev_data.get("ev", 0) + 1.0, 0.01)  # Approximate R from EV

        # Kelly Criterion
        if rr_ratio > 0:
            kelly = ((win_rate * (rr_ratio + 1)) - 1) / rr_ratio
        else:
            kelly = 0.0

        half_kelly = max(kelly * 0.5, 0.0)

        # Scale by Analyst's reliability weight
        adjusted_size = half_kelly * max(analyst_weight, 0.15)

        # Cap at 5% of capital
        final_size = min(adjusted_size, 0.05)

        return {
            "kelly_fraction": round(kelly, 4),
            "half_kelly": round(half_kelly, 4),
            "analyst_weight_adj": round(analyst_weight, 3),
            "recommended_size_pct": round(final_size * 100, 2),
            "interpretation": (
                f"Half-Kelly = {half_kelly:.2%} × Analyst Weight ({analyst_weight:.3f}) = "
                f"{final_size:.2%} of capital."
            ),
        }

    # ─────────────────────────────────────────
    # Main Analysis
    # ─────────────────────────────────────────

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Institutional-grade risk audit:
        1. Counter-Thesis generation (Why is this trade wrong?)
        2. Risk-to-Reward ratio (ATR-based)
        3. Expected Value calculation (using Analyst's historical win rate)
        4. Position sizing recommendation (Half-Kelly × Analyst weight)
        """
        df = context.ohlcv_df
        if df is None or df.empty or len(df) < 20:
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning="Insufficient data for institutional-grade risk analysis.",
            )

        # Get agent weights from context
        agent_weights = context.metadata.get("agent_weights", {})
        my_weight = agent_weights.get(self.name, None)
        analyst_weight = agent_weights.get("Analyst", 0.33)

        # Get Analyst's EMA win rate for EV calculation
        analyst_perf = context.metadata.get("agent_performance", {})
        analyst_win_rate = analyst_perf.get("Analyst", {}).get("ema_win_15m", 0.5)

        weight_note = ""
        if my_weight is not None:
            weight_note = (
                f" [My Reliability Weight: {my_weight:.3f} — "
                f"I must maximize this by providing accurate risk assessments with positive 15m markouts.]"
            )

        # ── Module 1: Counter-Thesis ──
        counter_thesis = self._generate_counter_thesis(df)

        # ── Module 2: Risk-to-Reward ──
        rr = self._calculate_risk_reward(df)

        # ── Module 3: Expected Value ──
        ev = self._calculate_expected_value(rr, analyst_win_rate)

        # ── Module 4: Position Sizing ──
        sizing = self._suggest_position_size(ev, analyst_weight)

        # ── Scoring: Higher score = more risk ──
        risk_score = len(counter_thesis)
        if not ev["ev_positive"]:
            risk_score += 2  # Negative EV is a major red flag

        # Determine signal
        signal = "HOLD"
        confidence = 0.5

        if risk_score >= 4:
            signal = "SELL" if df.iloc[-1].get("Momentum_5", 0) > 0 else "HOLD"
            confidence = min(0.55 + (risk_score * 0.08), 0.9)
        elif risk_score <= 1 and ev["ev_positive"]:
            signal = "BUY"
            confidence = min(0.55 + (ev["ev"] * 0.3), 0.85)

        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"COUNTER-THESIS: {'; '.join(counter_thesis)}")
        reasoning_parts.append(f"R:R = {rr['rr_ratio']}:1 (SL: {rr['stop_loss']:.4f}, TP: {rr['take_profit']:.4f}, Risk: {rr['risk_pct']:.2f}%)")
        reasoning_parts.append(f"EV: {ev['interpretation']}")
        reasoning_parts.append(f"POSITION SIZE: {sizing['interpretation']}")
        if weight_note:
            reasoning_parts.append(weight_note)

        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            metadata={
                "objections_count": len(counter_thesis),
                "counter_thesis": counter_thesis,
                "risk_reward": rr,
                "expected_value": ev,
                "position_sizing": sizing,
            },
        )

    def challenge(self, technician_signal: str, context: AgentContext) -> List[str]:
        """
        Generates institutional-grade objections to the Analyst's thesis.
        """
        df = context.ohlcv_df
        if df is None or df.empty:
            return ["No data to verify — cannot validate the trade thesis."]

        return self._generate_counter_thesis(df)
