import logging
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext
from core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class NarratorAgent(BaseAgent):
    def __init__(self, gemini_client: GeminiClient = None):
        self.name = "Sentiment Strategist"
        self.gemini_client = gemini_client or GeminiClient()

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Uses Gemini Flash for news sentiment analysis.
        """
        headlines = context.metadata.get("news_headlines", [])
        
        if not headlines:
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning="No news headlines provided to analyze.",
                metadata={}
            )
            
        try:
            sentiment_data = self.gemini_client.analyze_sentiment(headlines)
            
            return AgentOutput(
                agent_name=self.name,
                signal=sentiment_data.get("signal", "HOLD"),
                confidence=float(sentiment_data.get("confidence", 0.5)),
                reasoning=sentiment_data.get("reasoning", "Sentiment analysis fallback."),
                metadata={"headlines_count": len(headlines)}
            )
            
        except Exception as e:
            logger.error(f"Narrator agent failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning=f"Sentiment API error: {e}",
                metadata={}
            )
