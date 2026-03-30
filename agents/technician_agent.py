import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext

logger = logging.getLogger(__name__)


class TechnicianAgent(BaseAgent):
    """
    Institutional-Grade Technical Analyst.
    Employs Multi-Timeframe Analysis (MTA), Market Structure detection (BoS/ChoCH),
    Volume Profile / Order Flow analysis, and Liquidity Gap identification.
    Aware of its own Reliability Weight from Jackson's PerformanceTracker.
    """

    def __init__(self):
        self.name = "Analyst"

    # ─────────────────────────────────────────
    # Market Structure Detection
    # ─────────────────────────────────────────

    def _detect_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifies Higher-Highs (HH), Higher-Lows (HL), Lower-Highs (LH), Lower-Lows (LL).
        Detects Break of Structure (BoS) and Change of Character (ChoCH).
        """
        lookback = min(20, len(df) - 1)
        recent = df.tail(lookback)

        highs = recent["High"].values
        lows = recent["Low"].values

        # Swing detection using 5-bar pivot
        swing_highs = []
        swing_lows = []
        for i in range(2, len(recent) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                swing_lows.append((i, lows[i]))

        structure = {"trend": "NEUTRAL", "event": None, "swing_highs": len(swing_highs), "swing_lows": len(swing_lows)}

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1][1] > swing_highs[-2][1]  # Higher High
            hl = swing_lows[-1][1] > swing_lows[-2][1]     # Higher Low
            lh = swing_highs[-1][1] < swing_highs[-2][1]   # Lower High
            ll = swing_lows[-1][1] < swing_lows[-2][1]     # Lower Low

            if hh and hl:
                structure["trend"] = "BULLISH"
            elif lh and ll:
                structure["trend"] = "BEARISH"

            # Break of Structure: price closes beyond the last swing
            latest_close = df.iloc[-1]["Close"]
            if structure["trend"] == "BULLISH" and latest_close < swing_lows[-1][1]:
                structure["event"] = "BoS_BEARISH"  # Bullish structure broken downward
            elif structure["trend"] == "BEARISH" and latest_close > swing_highs[-1][1]:
                structure["event"] = "BoS_BULLISH"  # Bearish structure broken upward

            # Change of Character: opposite swing pattern forms
            if hh and ll:
                structure["event"] = "ChoCH"  # Mixed signals = character change

        return structure

    # ─────────────────────────────────────────
    # Multi-Timeframe Confluence (MTA)
    # ─────────────────────────────────────────

    def _multi_timeframe_confluence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compares short-term, medium-term, and long-term momentum for alignment.
        Uses Momentum_5 (short), Momentum (10, medium), Momentum_20 (long).
        """
        latest = df.iloc[-1]
        mtf = {"aligned": False, "direction": "NEUTRAL", "strength": 0.0}

        short = latest.get("Momentum_5", 0.0)
        medium = latest.get("Momentum", 0.0)
        long_term = latest.get("Momentum_20", 0.0)

        # Count how many timeframes agree
        bull_count = sum([1 for m in [short, medium, long_term] if m > 0])
        bear_count = sum([1 for m in [short, medium, long_term] if m < 0])

        if bull_count == 3:
            mtf["aligned"] = True
            mtf["direction"] = "BULLISH"
            mtf["strength"] = min(abs(short) + abs(medium) + abs(long_term), 1.0)
        elif bear_count == 3:
            mtf["aligned"] = True
            mtf["direction"] = "BEARISH"
            mtf["strength"] = min(abs(short) + abs(medium) + abs(long_term), 1.0)
        else:
            mtf["direction"] = "MIXED"
            mtf["strength"] = abs(bull_count - bear_count) / 3.0

        return mtf

    # ─────────────────────────────────────────
    # Volume Profile & Order Flow
    # ─────────────────────────────────────────

    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects institutional accumulation/distribution using VWAP + Volume analysis.
        """
        latest = df.iloc[-1]
        flow = {"bias": "NEUTRAL", "vwap_position": "AT", "volume_anomaly": False, "detail": ""}

        close = latest.get("Close", 0)
        vwap = latest.get("VWAP", 0)
        vol_ratio = latest.get("Volume_Ratio", 1.0)

        if vwap > 0:
            vwap_dist = (close - vwap) / vwap
            if vwap_dist > 0.005:
                flow["vwap_position"] = "ABOVE"
                flow["bias"] = "ACCUMULATION"
                flow["detail"] = f"Price {vwap_dist:.2%} above VWAP — institutional buying pressure."
            elif vwap_dist < -0.005:
                flow["vwap_position"] = "BELOW"
                flow["bias"] = "DISTRIBUTION"
                flow["detail"] = f"Price {abs(vwap_dist):.2%} below VWAP — institutional selling pressure."
            else:
                flow["detail"] = "Price at VWAP equilibrium — no clear institutional bias."

        if vol_ratio > 2.0:
            flow["volume_anomaly"] = True
            candle_dir = "bullish" if latest.get("Is_Bullish", 0) == 1 else "bearish"
            flow["detail"] += f" Significant {candle_dir} volume anomaly ({vol_ratio:.1f}x avg)."
        elif vol_ratio < 0.5:
            flow["detail"] += f" Thin volume ({vol_ratio:.1f}x avg) — low conviction move."

        return flow

    # ─────────────────────────────────────────
    # Liquidity Gap Detection
    # ─────────────────────────────────────────

    def _detect_liquidity_gaps(self, df: pd.DataFrame) -> List[str]:
        """
        Identifies unfilled price gaps between consecutive candles.
        These gaps often act as magnets where price tends to return.
        """
        gaps = []
        lookback = min(10, len(df) - 1)
        recent = df.tail(lookback)

        for i in range(1, len(recent)):
            prev_close = recent.iloc[i - 1]["Close"]
            curr_open = recent.iloc[i]["Open"]
            gap_pct = abs(curr_open - prev_close) / prev_close

            if gap_pct > 0.003:  # > 0.3% gap
                direction = "up" if curr_open > prev_close else "down"
                gaps.append(
                    f"Liquidity gap {direction} at {prev_close:.4f}→{curr_open:.4f} ({gap_pct:.2%})"
                )

        return gaps[-3:]  # Return most recent 3 gaps

    # ─────────────────────────────────────────
    # High-Probability Entry Zone
    # ─────────────────────────────────────────

    def _calculate_entry_zone(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates a high-probability entry zone using VWAP ± 0.5×ATR.
        """
        latest = df.iloc[-1]
        vwap = latest.get("VWAP", latest["Close"])
        atr = latest.get("ATR", 0)

        return {
            "zone_low": round(vwap - 0.5 * atr, 6),
            "zone_high": round(vwap + 0.5 * atr, 6),
            "vwap": round(vwap, 6),
            "atr": round(atr, 6),
        }

    # ─────────────────────────────────────────
    # Main Analysis
    # ─────────────────────────────────────────

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Institutional-grade technical analysis:
        1. Market Structure (BoS / ChoCH)
        2. Multi-Timeframe Confluence
        3. Order Flow (VWAP + Volume)
        4. Liquidity Gap Detection
        5. High-Probability Entry Zone
        """
        df = context.ohlcv_df
        if df is None or df.empty or len(df) < 50:
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.0,
                reasoning="Insufficient data for institutional-grade technical analysis.",
                trigger_price=None,
                valid_until=None,
            )

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Get own reliability weight from context
        my_weight = context.metadata.get("agent_weights", {}).get(self.name, None)
        weight_note = ""
        if my_weight is not None:
            weight_note = f" [My Reliability Weight: {my_weight:.3f} — I must maximize this by providing accurate signals with positive 15m markouts.]"

        # ── Module 1: Market Structure ──
        structure = self._detect_market_structure(df)

        # ── Module 2: Multi-Timeframe Confluence ──
        mtf = self._multi_timeframe_confluence(df)

        # ── Module 3: Order Flow ──
        order_flow = self._analyze_order_flow(df)

        # ── Module 4: Liquidity Gaps ──
        gaps = self._detect_liquidity_gaps(df)

        # ── Module 5: Entry Zone ──
        entry_zone = self._calculate_entry_zone(df)

        # ── Scoring Engine ──
        score = 0.0
        thesis_points = []

        # Market Structure contribution (+/-2)
        if structure["trend"] == "BULLISH":
            score += 1.5
            thesis_points.append(f"Bullish market structure (HH/HL pattern)")
        elif structure["trend"] == "BEARISH":
            score -= 1.5
            thesis_points.append(f"Bearish market structure (LH/LL pattern)")

        if structure["event"] == "BoS_BULLISH":
            score += 2.0
            thesis_points.append("Break of Structure — bullish reversal detected")
        elif structure["event"] == "BoS_BEARISH":
            score -= 2.0
            thesis_points.append("Break of Structure — bearish breakdown detected")
        elif structure["event"] == "ChoCH":
            thesis_points.append("Change of Character — trend transition in progress")

        # MTA contribution (+/-1.5)
        if mtf["aligned"]:
            if mtf["direction"] == "BULLISH":
                score += 1.5
                thesis_points.append(f"Multi-Timeframe Confluence: all timeframes bullish (strength {mtf['strength']:.2f})")
            elif mtf["direction"] == "BEARISH":
                score -= 1.5
                thesis_points.append(f"Multi-Timeframe Confluence: all timeframes bearish (strength {mtf['strength']:.2f})")
        else:
            thesis_points.append(f"MTA divergence: timeframes not aligned ({mtf['direction']})")

        # Order Flow contribution (+/-1)
        if order_flow["bias"] == "ACCUMULATION":
            score += 1.0
            thesis_points.append(f"Order Flow: {order_flow['detail']}")
        elif order_flow["bias"] == "DISTRIBUTION":
            score -= 1.0
            thesis_points.append(f"Order Flow: {order_flow['detail']}")

        if order_flow["volume_anomaly"]:
            vol_dir = 0.5 if latest.get("Is_Bullish", 0) == 1 else -0.5
            score += vol_dir
            thesis_points.append(f"Volume anomaly detected ({order_flow['detail'].split('.')[-2].strip()})")

        # Classic indicator confirmation (lighter weight in institutional model)
        if "RSI" in latest:
            rsi = latest["RSI"]
            if rsi < 30:
                score += 0.5
                thesis_points.append(f"RSI oversold ({rsi:.1f}) — mean reversion potential")
            elif rsi > 70:
                score -= 0.5
                thesis_points.append(f"RSI overbought ({rsi:.1f}) — exhaustion risk")

        if "MACD" in latest and "MACD_Signal" in latest:
            if prev["MACD"] < prev["MACD_Signal"] and latest["MACD"] > latest["MACD_Signal"]:
                score += 0.5
                thesis_points.append("MACD bullish crossover confirmation")
            elif prev["MACD"] > prev["MACD_Signal"] and latest["MACD"] < latest["MACD_Signal"]:
                score -= 0.5
                thesis_points.append("MACD bearish crossover confirmation")

        # Liquidity Gaps
        if gaps:
            thesis_points.append(f"Liquidity gaps: {'; '.join(gaps)}")

        # ── Performance-Based Calibration ──
        # Anchor heuristic confidence to historical win rate (to fix overconfidence)
        calibration_note = ""
        perf = context.metadata.get("agent_performance", {}).get(self.name, {})
        win_rate = perf.get("ema_win_15m", 0.5)
        n_resolved = perf.get("n_resolved", 0)
        
        # ── Final Signal ──
        if score >= 2.0:
            signal = "BUY"
            confidence = min(0.55 + (score * 0.06), 0.95)
        elif score <= -2.0:
            signal = "SELL"
            confidence = min(0.55 + (abs(score) * 0.06), 0.95)
        else:
            signal = "HOLD"
            confidence = 0.5

        if signal != "HOLD":
            # If we have enough samples, adjust confidence toward the win rate
            if n_resolved >= 5:
                # Weighted blend: 40% heuristic, 60% historical reliability
                calibrated_conf = (confidence * 0.4) + (win_rate * 0.6)
                
                # If win rate is poor (<50%), apply a 'skepticism penalty'
                if win_rate < 0.5:
                    calibrated_conf *= 0.85 
                    calibration_note = f" [Calibration: Confidence deflated due to {win_rate:.1%} historical reliability]."
                else:
                    calibration_note = f" [Calibration: Performance-anchored at {calibrated_conf:.1%}]."
                
                confidence = min(max(calibrated_conf, 0.3), 0.95)

        if not thesis_points:
            thesis_points.append("No high-conviction institutional setups identified.")

        # Build Technical Thesis
        entry_str = f"High-Probability Entry Zone: {entry_zone['zone_low']:.4f} – {entry_zone['zone_high']:.4f} (VWAP ± 0.5×ATR)"
        thesis = f"TECHNICAL THESIS: {'; '.join(thesis_points)}. {entry_str}.{weight_note}{calibration_note}"

        trigger_price = latest.get("Close", 0)

        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=thesis,
            metadata={
                "score": score,
                "market_structure": structure,
                "mtf": mtf,
                "order_flow": order_flow,
                "entry_zone": entry_zone,
                "gaps": gaps,
            },
            trigger_price=trigger_price,
            valid_until=None,
        )

    def rebuttal(self, critic_objections: List[str], df: pd.DataFrame) -> str:
        """
        Institutional-grade rebuttal using Order Flow and Market Structure evidence.
        """
        if df is None or df.empty:
            return "Unable to formulate a rebuttal due to lack of market data."

        latest = df.iloc[-1]
        rebuttal_points = []

        # Market Structure defense
        structure = self._detect_market_structure(df)
        if structure["trend"] == "BULLISH":
            rebuttal_points.append(f"Market structure remains intact (bullish HH/HL). No structural breakdown has occurred.")
        elif structure["trend"] == "BEARISH":
            rebuttal_points.append(f"Bearish structure (LH/LL) supports the directional thesis.")

        # VWAP defense
        if "VWAP" in latest:
            vwap_pos = "above" if latest["Close"] > latest["VWAP"] else "below"
            rebuttal_points.append(
                f"Price sustaining {vwap_pos} VWAP ({latest['VWAP']:.4f}) confirms institutional order flow direction."
            )

        # Volatility context
        if "ATR" in latest and "ATR" in df.columns:
            atr_avg = df["ATR"].rolling(14).mean().iloc[-1]
            atr_trend = "expanding" if latest["ATR"] > atr_avg else "contracting"
            rebuttal_points.append(
                f"Volatility is {atr_trend} (ATR: {latest['ATR']:.4f} vs 14-period avg: {atr_avg:.4f}), consistent with directional conviction."
            )

        # MTA defense
        mtf = self._multi_timeframe_confluence(df)
        if mtf["aligned"]:
            rebuttal_points.append(
                f"Multi-timeframe alignment ({mtf['direction']}, strength {mtf['strength']:.2f}) provides structural confluence."
            )

        if not rebuttal_points:
            return "The technical structure is fragile; the critic's points are valid."

        return "ANALYST REBUTTAL: " + " Furthermore, ".join(rebuttal_points)
