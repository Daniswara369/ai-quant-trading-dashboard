import pandas as pd
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext

class TechnicianAgent(BaseAgent):
    def __init__(self):
        self.name = "Analyst"

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Rules-based technical analysis looking for trend and momentum signals.
        """
        df = context.ohlcv_df
        if df is None or df.empty or len(df) < 50:
             return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.0,
                reasoning="Insufficient data for technical analysis.",
                trigger_price=None,
                valid_until=None
            )
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        reasons = []
        
        # 1. MACD Crossover
        if "MACD" in latest and "MACD_Signal" in latest:
            if prev["MACD"] < prev["MACD_Signal"] and latest["MACD"] > latest["MACD_Signal"]:
                score += 1
                reasons.append("Bullish MACD crossover")
            elif prev["MACD"] > prev["MACD_Signal"] and latest["MACD"] < latest["MACD_Signal"]:
                score -= 1
                reasons.append("Bearish MACD crossover")
                
        # 2. RSI Extremes
        if "RSI" in latest:
            rsi = latest["RSI"]
            if rsi < 30:
                score += 1
                reasons.append(f"Oversold RSI ({rsi:.1f})")
            elif rsi > 70:
                score -= 1
                reasons.append(f"Overbought RSI ({rsi:.1f})")
                
        # 3. Golden/Death Cross (SMA 50 vs SMA 200 - simplify to 20 vs 50 if using short term in config)
        if "SMA_20" in latest and "SMA_50" in latest:
            if prev["SMA_20"] <= prev["SMA_50"] and latest["SMA_20"] > latest["SMA_50"]:
                score += 1.5
                reasons.append("Golden Cross (SMA20 > SMA50)")
            elif prev["SMA_20"] >= prev["SMA_50"] and latest["SMA_20"] < latest["SMA_50"]:
                score -= 1.5
                reasons.append("Death Cross (SMA20 < SMA50)")
                
        # 4. Bollinger Squeeze & Breakout
        if "BB_Pct" in latest and "BB_Width" in latest:
            if latest["BB_Width"] < df["BB_Width"].rolling(20).mean().iloc[-1]: # squeeze
                if latest["Close"] > latest["BB_Upper"]:
                    score += 1
                    reasons.append("Bullish Bollinger Band breakout from squeeze")
                elif latest["Close"] < latest["BB_Lower"]:
                    score -= 1
                    reasons.append("Bearish Bollinger Band breakdown from squeeze")
                    
        # 5. Volume Anomaly
        if "Volume_Ratio" in latest and latest["Volume_Ratio"] > 2.0:
            if latest["Close"] > latest["Open"]:
                score += 0.5
                reasons.append("Significant bullish volume anomaly")
            else:
                score -= 0.5
                reasons.append("Significant bearish volume anomaly")

        # Determine signal and confidence
        if score >= 1.5:
            signal = "BUY"
            confidence = min(0.5 + (score * 0.1), 0.95)
        elif score <= -1.5:
            signal = "SELL"
            confidence = min(0.5 + (abs(score) * 0.1), 0.95)
        else:
            signal = "HOLD"
            confidence = 0.5
            
        if not reasons:
            reasons.append("No strong technical signals generated.")
            
        reasoning_str = "; ".join(reasons)
        
        # Valid until next candle (assuming 1h)
        # We don't have timedelta here easily without knowing timeframe exact logic, 
        # but leaving it abstract.
        trigger_price = latest.get("Close", 0)
        
        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning_str,
            metadata={"score": score},
            trigger_price=trigger_price,
            valid_until=None
        )

    def rebuttal(self, critic_objections: List[str], df: pd.DataFrame) -> str:
        """
        Provides counter-arguments to the critic using secondary indicators like Volume Profile, ATR.
        """
        if df is None or df.empty:
            return "Unable to formulate a rebuttal due to lack of market data."
            
        latest = df.iloc[-1]
        rebuttal_points = []
        
        # Look for supporting evidence for trend continuation
        if "ATR" in latest and "ATR" in df.columns:
            atr_trend = "expanding" if latest["ATR"] > df["ATR"].rolling(14).mean().iloc[-1] else "contracting"
            rebuttal_points.append(f"Volatility is {atr_trend} (ATR: {latest['ATR']:.4f}), which may support the move.")
            
        if "VWAP" in latest and latest["Close"] > latest["VWAP"]:
            rebuttal_points.append(f"Price is sustaining above VWAP ({latest['VWAP']:.4f}), indicating institutional support.")
        elif "VWAP" in latest and latest["Close"] < latest["VWAP"]:
            rebuttal_points.append(f"Price is heavily rejected below VWAP ({latest['VWAP']:.4f}), indicating overhead supply.")
            
        if not rebuttal_points:
            return "The technical structure is fragile; the critic's points are valid."
            
        return "Technician Rebuttal: " + " Furthermore, ".join(rebuttal_points)
