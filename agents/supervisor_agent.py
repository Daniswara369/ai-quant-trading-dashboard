import logging
from typing import Dict, Any, List
from agents.technician_agent import TechnicianAgent
from agents.narrator_agent import NarratorAgent
from agents.critic_agent import CriticAgent
from core.schemas import AgentOutput, AgentContext, WeightedConsensus, MarketRegime
from core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self, gemini_client: GeminiClient = None):
        self.name = "Manager"
        self.gemini_client = gemini_client or GeminiClient()
        self.technician = TechnicianAgent()
        self.narrator = NarratorAgent(self.gemini_client)
        self.critic = CriticAgent()
        
    def determine_regime(self, context: AgentContext) -> MarketRegime:
        """Determines the market regime based on context to assign dynamic weights."""
        df = context.ohlcv_df
        if df is None or df.empty:
            return MarketRegime(regime="Default", confidence=0.5, reasoning="No data to determine regime.")
            
        latest = df.iloc[-1]
        regime_label = latest.get("Regime_Label", "Sideways")
        
        # Check for News-Volatile
        is_volatile = False
        if "ATR" in latest and "ATR" in df.columns:
            if latest["ATR"] > df["ATR"].rolling(14).mean().iloc[-1] * 1.5:
                is_volatile = True
                
        news_count = len(context.metadata.get("news_headlines", []))
        
        if is_volatile and news_count > 0:
            return MarketRegime(regime="News-Volatile", confidence=0.8, reasoning="High ATR volatility with active news.")
        elif regime_label in ["Bull", "Bear"]:
            return MarketRegime(regime="Trending", confidence=0.7, reasoning=f"Strong {regime_label} trend detected.")
        else:
            return MarketRegime(regime="Ranging", confidence=0.6, reasoning="Sideways market structure.")

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Returns the specific agent weights based on the regime."""
        weights = {
            "Trending": {"Analyst": 0.60, "Sentiment Strategist": 0.15, "Risk Auditor": 0.25},
            "Ranging": {"Analyst": 0.20, "Sentiment Strategist": 0.30, "Risk Auditor": 0.50},
            "News-Volatile": {"Analyst": 0.10, "Sentiment Strategist": 0.70, "Risk Auditor": 0.20},
            "Default": {"Analyst": 0.40, "Sentiment Strategist": 0.25, "Risk Auditor": 0.35}
        }
        return weights.get(regime, weights["Default"])
        
    def analyze_and_debate(self, context: AgentContext) -> WeightedConsensus:
        """
        Executes the adversarial debate loop:
        1. Gets signals from all agents.
        2. Supervisor triggers debate synthesis via Gemini Pro.
        3. Applies dynamic regime weights.
        """
        # Step 1: Detect Regime
        market_regime = self.determine_regime(context)
        weights = self.get_regime_weights(market_regime.regime)
        
        # Step 2: Agent Analyses
        tech_out = self.technician.analyze(context)
        narr_out = self.narrator.analyze(context)
        
        # Critic challenges the Technician
        critic_out = self.critic.analyze(context)
        critic_objections = self.critic.challenge(tech_out.signal, context)
        
        # Technician Rebuttal
        tech_rebuttal = self.technician.rebuttal(critic_objections, context.ohlcv_df)
        
        # Step 3: Synthesis Debate via Gemini
        debate_result = self.gemini_client.supervisor_debate(
            technician_signal=f"Signal: {tech_out.signal}, Confidence: {tech_out.confidence}, Reason: {tech_out.reasoning}",
            technician_rebuttal=tech_rebuttal,
            critic_objections=critic_objections,
            context={"symbol": context.symbol, "regime": market_regime.regime}
        )
        
        # Base logic for numerical aggregation
        # Convert signals to numeric
        def to_numeric(signal_str: str) -> int:
             return 1 if signal_str == "BUY" else (-1 if signal_str == "SELL" else 0)
             
        tech_score = to_numeric(tech_out.signal) * tech_out.confidence * weights["Analyst"]
        narr_score = to_numeric(narr_out.signal) * narr_out.confidence * weights["Sentiment Strategist"]
        critic_score = to_numeric(critic_out.signal) * critic_out.confidence * weights["Risk Auditor"]
        
        raw_weighted_score = tech_score + narr_score + critic_score
        base_confidence = abs(raw_weighted_score)
        
        # Apply Gemini's confidence adjustment (e.g., -0.2 to +0.2)
        confidence_adj = float(debate_result.get("confidence_adj", 0.0))
        final_confidence = min(max(base_confidence + confidence_adj, 0.0), 1.0)
        
        if raw_weighted_score > 0.1:
            final_signal = "BUY"
        elif raw_weighted_score < -0.1:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
            
        return WeightedConsensus(
            signal=final_signal,
            confidence=final_confidence,
            reasoning=debate_result.get("reasoning", "Debate failed; using base technical weighting."),
            agent_outputs=[tech_out, narr_out, critic_out],
            regime=market_regime
        )
