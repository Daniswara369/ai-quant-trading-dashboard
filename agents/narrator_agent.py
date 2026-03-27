import logging
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext
from core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class NarratorAgent(BaseAgent):
    """
    Institutional-Grade Sentiment & Macro Strategist.
    Detects Risk-On / Risk-Off market regimes, identifies Sentiment Divergence,
    and rates Narrative Strength. Uses Gemini for advanced interpretation.
    Aware of its own Reliability Weight from Jackson's PerformanceTracker.
    """

    def __init__(self, gemini_client: GeminiClient = None):
        self.name = "Sentiment Strategist"
        self.gemini_client = gemini_client or GeminiClient()

    def _detect_sentiment_divergence(self, context: AgentContext) -> Dict[str, Any]:
        """
        Compares headline sentiment bias vs price momentum.
        Divergence = headlines bullish but price falling, or vice versa.
        """
        df = context.ohlcv_df
        divergence = {"detected": False, "type": None, "detail": ""}

        if df is None or df.empty:
            return divergence

        latest = df.iloc[-1]
        momentum_5 = latest.get("Momentum_5", 0.0)

        headlines = context.metadata.get("news_headlines", [])
        if not headlines:
            return divergence

        # Simple heuristic: count bullish/bearish keywords in headlines
        bull_words = {"surge", "rally", "gain", "bullish", "up", "rise", "high", "record", "buy", "growth", "positive"}
        bear_words = {"crash", "drop", "fall", "bearish", "down", "decline", "low", "sell", "loss", "fear", "negative"}

        bull_count = sum(1 for h in headlines for w in bull_words if w in h.lower())
        bear_count = sum(1 for h in headlines for w in bear_words if w in h.lower())

        headline_bias = "BULLISH" if bull_count > bear_count else ("BEARISH" if bear_count > bull_count else "NEUTRAL")
        price_bias = "BULLISH" if momentum_5 > 0.005 else ("BEARISH" if momentum_5 < -0.005 else "NEUTRAL")

        if headline_bias == "BULLISH" and price_bias == "BEARISH":
            divergence["detected"] = True
            divergence["type"] = "BEARISH_DIVERGENCE"
            divergence["detail"] = "Headlines bullish but price momentum is negative — smart money may be distributing."
        elif headline_bias == "BEARISH" and price_bias == "BULLISH":
            divergence["detected"] = True
            divergence["type"] = "BULLISH_DIVERGENCE"
            divergence["detail"] = "Headlines bearish but price momentum is positive — smart money may be accumulating against the crowd."

        return divergence

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Institutional-grade macro & sentiment analysis:
        1. Risk-On / Risk-Off regime detection (via Gemini)
        2. Sentiment Divergence (headlines vs price momentum)
        3. Narrative Strength rating
        """
        headlines = context.metadata.get("news_headlines", [])

        # Get own reliability weight from context
        my_weight = context.metadata.get("agent_weights", {}).get(self.name, None)
        weight_instruction = ""
        if my_weight is not None:
            weight_instruction = (
                f"\n\nPERFORMANCE CONTEXT: Your current Reliability Weight is {my_weight:.3f}. "
                f"Your objective is to maximize this weight by providing signals that result in "
                f"positive 15-minute markouts. Be precise and avoid false signals."
            )

        if not headlines:
            # Even without headlines, analyze price-based regime
            df = context.ohlcv_df
            regime_note = "No news headlines available."
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                vol = latest.get("Rolling_Volatility", 0)
                momentum = latest.get("Momentum_20", 0)
                if vol > 0.03 and momentum < -0.05:
                    regime_note += " Price regime suggests Risk-Off (high volatility + negative momentum)."
                elif vol < 0.015 and momentum > 0.02:
                    regime_note += " Price regime suggests Risk-On (low volatility + positive momentum)."
                else:
                    regime_note += " Price regime is ambiguous — no clear Risk-On/Off signal."

            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning=regime_note + (f" {weight_instruction.strip()}" if weight_instruction else ""),
                metadata={"regime": "AMBIGUOUS", "narrative_strength": 0},
            )

        # Detect sentiment divergence before calling Gemini
        divergence = self._detect_sentiment_divergence(context)

        # Build the institutional-grade prompt for Gemini
        divergence_context = ""
        if divergence["detected"]:
            divergence_context = f"""
            
            SENTIMENT DIVERGENCE ALERT: {divergence['detail']}
            Factor this divergence heavily into your analysis. Divergence between 
            sentiment and price action often precedes major reversals."""

        try:
            import json
            model_obj = self.gemini_client
            prompt = f"""You are an institutional macro strategist at a top-tier hedge fund.
            
            Analyze the following news headlines for a trading asset and provide:
            
            1. **Market Regime**: Classify as "RISK_ON" or "RISK_OFF" based on the macro narrative.
               - Risk-On: Markets favor growth assets, risk appetite is high.
               - Risk-Off: Markets favor safe havens, fear dominates.
            
            2. **Narrative Strength**: Rate from 1-10 how strong/unified the current news narrative is.
               - 10 = Every headline points the same direction with high conviction.
               - 1 = Headlines are contradictory or irrelevant.
            
            3. **Signal**: Based on your regime analysis, output BUY (Risk-On favoring the asset), 
               SELL (Risk-Off threatening the asset), or HOLD (ambiguous).
            
            4. **Confidence**: A float 0.0 to 1.0 reflecting how confident you are.
            
            5. **Reasoning**: A brief institutional-quality explanation referencing specific headlines.
            {divergence_context}
            {weight_instruction}
            
            Headlines:
            {json.dumps(headlines)}
            
            Respond strictly in valid JSON format:
            {{
                "regime": "RISK_ON" or "RISK_OFF" or "AMBIGUOUS",
                "narrative_strength": <int 1-10>,
                "signal": "BUY" or "SELL" or "HOLD",
                "confidence": <float 0.0-1.0>,
                "reasoning": "<institutional-quality explanation>"
            }}"""

            # Use the existing analyze_sentiment infrastructure but with our custom prompt
            import google.generativeai as genai

            model = genai.GenerativeModel(model_obj.flash_model)
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            result = json.loads(text.strip())

            # Adjust confidence if divergence detected
            signal = result.get("signal", "HOLD")
            confidence = float(result.get("confidence", 0.5))
            regime = result.get("regime", "AMBIGUOUS")
            narrative_strength = int(result.get("narrative_strength", 5))

            reasoning_parts = [result.get("reasoning", "Sentiment analysis fallback.")]

            if divergence["detected"]:
                reasoning_parts.append(f"⚠️ DIVERGENCE: {divergence['detail']}")
                # Divergence reduces confidence
                confidence = max(confidence - 0.15, 0.3)

            reasoning_parts.append(f"Regime: {regime} | Narrative Strength: {narrative_strength}/10")

            return AgentOutput(
                agent_name=self.name,
                signal=signal,
                confidence=confidence,
                reasoning=" | ".join(reasoning_parts),
                metadata={
                    "headlines_count": len(headlines),
                    "regime": regime,
                    "narrative_strength": narrative_strength,
                    "divergence": divergence,
                },
            )

        except Exception as e:
            logger.error(f"Sentiment Strategist failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning=f"Sentiment API error: {e}",
                metadata={},
            )
