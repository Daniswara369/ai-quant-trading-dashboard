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
        """Returns the specific agent weights based on the regime or dynamic engine."""
        try:
            from config import AGENT_SIGNAL_LOG_PATH, AGENT_COMMITTEE_JSON
            import json
            from agent_consensus.signal_log import SignalLogStore
            from agent_consensus.snapshot import build_committee_snapshot

            store = SignalLogStore(AGENT_SIGNAL_LOG_PATH)
            committee_names = json.loads(AGENT_COMMITTEE_JSON)
            
            # If the log actually has data, we use the dynamic snapshot
            if len(store.load_all()) > 0:
                snapshot = build_committee_snapshot(store, committee_names)
                dynamic_weights = snapshot.get("weights", {})
                
                if dynamic_weights:
                    logger.info(f"Supervisor overriding {regime} with dynamic EMA weights: {dynamic_weights}")
                    return dynamic_weights
        except Exception as e:
            logger.warning(f"Failed to load dynamic committee snapshot, falling back: {e}")

        # Fallback static weights
        weights = {
            "Trending": {"Analyst": 0.60, "Sentiment Strategist": 0.15, "Risk Auditor": 0.25},
            "Ranging": {"Analyst": 0.20, "Sentiment Strategist": 0.30, "Risk Auditor": 0.50},
            "News-Volatile": {"Analyst": 0.10, "Sentiment Strategist": 0.70, "Risk Auditor": 0.20},
            "Default": {"Analyst": 0.40, "Sentiment Strategist": 0.25, "Risk Auditor": 0.35}
        }
        logger.info(f"Supervisor using fallback static weights for {regime}")
        return weights.get(regime, weights["Default"])
        
    def _build_committee_context(self, snapshot: Dict) -> str:
        """Build a human-readable committee context string for the Gemini prompt."""
        lines = []
        weights = snapshot.get("weights", {})
        perf_list = snapshot.get("performance", [])
        perf_by_name = {p["agent_name"]: p for p in perf_list}
        
        for agent_name, weight in weights.items():
            perf = perf_by_name.get(agent_name, {})
            win_rate = perf.get("ema_win_15m", 0.0)
            n_resolved = perf.get("n_resolved", 0)
            lines.append(
                f"- {agent_name}: Reliability Weight = {weight:.3f} "
                f"(15m Win Rate: {win_rate:.1%}, Trades Resolved: {n_resolved})"
            )
        
        return "\n".join(lines) if lines else ""

    def _log_signals_to_store(self, agent_outputs: List, context: AgentContext, entry_price: float):
        """Log each agent's signal to Jackson's SignalLogStore for future performance tracking."""
        try:
            import json
            import uuid
            from config import AGENT_SIGNAL_LOG_PATH
            from agent_consensus.signal_log import SignalLogStore
            from agent_consensus.schema import AgentVote

            store = SignalLogStore(AGENT_SIGNAL_LOG_PATH)
            decision_id = str(uuid.uuid4())[:8]

            signal_to_direction = {"BUY": 1, "SELL": -1, "HOLD": 0}

            for agent_out in agent_outputs:
                direction = signal_to_direction.get(agent_out.signal, 0)
                vote = AgentVote(
                    agent_name=agent_out.agent_name,
                    direction=direction,
                    confidence=agent_out.confidence,
                    entry_price=entry_price,
                    symbol=context.symbol,
                    decision_id=decision_id,
                )
                store.append_vote(vote)

            logger.info(f"Logged {len(agent_outputs)} agent signals to store (decision_id={decision_id})")
        except Exception as e:
            logger.warning(f"Failed to log signals to store: {e}")

    def analyze_and_debate(self, context: AgentContext) -> WeightedConsensus:
        """
        Executes the adversarial debate loop:
        1. Gets signals from all agents.
        2. Builds a committee snapshot and injects it into the Gemini prompt.
        3. Supervisor triggers debate synthesis via Gemini Pro.
        4. Applies dynamic regime weights.
        5. Logs all agent signals to Jackson's SignalLogStore.
        """
        # Step 1: Detect Regime & Load Dynamic Weights + Snapshot
        market_regime = self.determine_regime(context)
        weights = self.get_regime_weights(market_regime.regime)
        
        # Build committee context for prompt injection
        committee_context = ""
        try:
            import json as _json
            from config import AGENT_SIGNAL_LOG_PATH, AGENT_COMMITTEE_JSON
            from agent_consensus.signal_log import SignalLogStore
            from agent_consensus.snapshot import build_committee_snapshot

            store = SignalLogStore(AGENT_SIGNAL_LOG_PATH)
            committee_names = _json.loads(AGENT_COMMITTEE_JSON)
            rows = store.load_all()
            if len(rows) > 0:
                snapshot = build_committee_snapshot(store, committee_names)
                committee_context = self._build_committee_context(snapshot)
                logger.info(f"Committee context for Gemini prompt:\n{committee_context}")
        except Exception as e:
            logger.warning(f"Could not build committee context for prompt: {e}")
        
        # Inject agent weights and performance into context for the agents
        try:
            import json as _json2
            from config import AGENT_SIGNAL_LOG_PATH, AGENT_COMMITTEE_JSON
            from agent_consensus.signal_log import SignalLogStore
            from agent_consensus.snapshot import build_committee_snapshot

            store = SignalLogStore(AGENT_SIGNAL_LOG_PATH)
            committee_names = _json2.loads(AGENT_COMMITTEE_JSON)
            rows = store.load_all()
            if len(rows) > 0:
                snap = build_committee_snapshot(store, committee_names)
                context.metadata["agent_weights"] = snap.get("weights", {})
                # Build a quick lookup for agent performance
                perf_lookup = {}
                for p in snap.get("performance", []):
                    perf_lookup[p["agent_name"]] = p
                context.metadata["agent_performance"] = perf_lookup
                logger.info(f"Injected agent weights into context: {context.metadata['agent_weights']}")
        except Exception as e:
            logger.warning(f"Could not inject agent weights into context: {e}")

        # Step 2: Agent Analyses
        tech_out = self.technician.analyze(context)
        narr_out = self.narrator.analyze(context)
        
        # Critic challenges the Technician
        critic_out = self.critic.analyze(context)
        critic_objections = self.critic.challenge(tech_out.signal, context)
        
        # Technician Rebuttal
        tech_rebuttal = self.technician.rebuttal(critic_objections, context.ohlcv_df)
        
        # Step 3: Synthesis Debate via Gemini (with committee weights injected)
        debate_result = self.gemini_client.supervisor_debate(
            technician_signal=f"Signal: {tech_out.signal}, Confidence: {tech_out.confidence}, Reason: {tech_out.reasoning}",
            technician_rebuttal=tech_rebuttal,
            critic_objections=critic_objections,
            context={"symbol": context.symbol, "regime": market_regime.regime},
            committee_snapshot=committee_context,
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
        
        # Step 4: Log all agent signals to Jackson's store for future performance tracking
        entry_price = 0.0
        try:
            if context.ohlcv_df is not None and not context.ohlcv_df.empty:
                entry_price = float(context.ohlcv_df.iloc[-1].get("Close", 0.0))
        except Exception:
            pass
        self._log_signals_to_store([tech_out, narr_out, critic_out], context, entry_price)
            
        return WeightedConsensus(
            signal=final_signal,
            confidence=final_confidence,
            reasoning=debate_result.get("reasoning", "Debate failed; using base technical weighting."),
            agent_outputs=[tech_out, narr_out, critic_out],
            regime=market_regime
        )
