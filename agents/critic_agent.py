import pandas as pd
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext

class CriticAgent(BaseAgent):
    def __init__(self):
        self.name = "Risk Auditor"

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Rules-based skeptics looking for traps, divergence, false breakouts.
        Returns a counter-signal or HOLD if no traps found.
        """
        df = context.ohlcv_df
        if df is None or df.empty or len(df) < 20:
             return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning="Insufficient data for critical analysis."
            )
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        objections = self.challenge(None, context) # Generates all possible objections based on state
        
        score = len(objections)
        # The critic inherently is adversarial to trends.
        # So if we see bullish traps, we signal SELL. Let's just output HOLD if no objections.
        
        signal = "HOLD"
        confidence = 0.5
        reasoning_str = "No major traps identified."
        
        if objections:
            reasoning_str = "Found risks: " + "; ".join(objections)
            confidence = min(0.5 + (score * 0.1), 0.9)
            # If RSI is extremely overbought, we are looking for a SELL signal
            if "Overbought" in reasoning_str or "Bull Trap" in reasoning_str:
                signal = "SELL"
            elif "Oversold" in reasoning_str or "Bear Trap" in reasoning_str:
                signal = "BUY"
                
        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning_str,
            metadata={"objections_count": score}
        )

    def challenge(self, technician_signal: str, context: AgentContext) -> List[str]:
        """
        Explicitly generates <=3 reasons not to trade / trap indicators.
        """
        df = context.ohlcv_df
        if df is None or df.empty:
            return ["No data to verify."]
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        objections = []
        
        # 1. RSI Extremes
        if "RSI" in latest:
            if latest["RSI"] > 75:
                objections.append("Extreme Overbought RSI - potential reversal.")
            elif latest["RSI"] < 25:
                objections.append("Extreme Oversold RSI - potential knife catching.")
                
        # 2. Bull/Bear Trap (Price > BB_Upper but closing < Open)
        if "BB_Upper" in latest and "BB_Lower" in latest:
            if latest["High"] > latest["BB_Upper"] and latest["Close"] < prev["Close"]:
                objections.append("Possible Bull Trap (Failed breakout above upper Bollinger band).")
            elif latest["Low"] < latest["BB_Lower"] and latest["Close"] > prev["Close"]:
                objections.append("Possible Bear Trap (Failed breakdown below lower Bollinger band).")
                
        # 3. Volume Divergence
        if "Volume_Ratio" in latest and "Volume_MA" in latest:
            # Price making new high but volume is low
            if latest["Close"] > df["Close"].rolling(10).max().iloc[-2] and latest["Volume_Ratio"] < 0.8:
                objections.append("Price rising on low volume - weak structural support.")
            elif latest["Close"] < df["Close"].rolling(10).min().iloc[-2] and latest["Volume_Ratio"] < 0.8:
                objections.append("Price falling on low volume - weak selling pressure.")
                
        # Return top 3 objections
        return objections[:3]
