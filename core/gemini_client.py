import os
import json
import logging
import google.generativeai as genai
from typing import Dict, Any, List
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini features will be disabled or fail.")
        else:
            genai.configure(api_key=self.api_key)
            
        self.flash_model = GEMINI_FLASH_MODEL
        self.pro_model = GEMINI_PRO_MODEL
        
    def analyze_sentiment(self, headlines: List[str]) -> Dict[str, Any]:
        """Uses Gemini Flash for news sentiment analysis. Returns {signal, confidence, reasoning}."""
        if not self.api_key:
            return {"signal": "HOLD", "confidence": 0.5, "reasoning": "API Key missing."}
            
        try:
            model = genai.GenerativeModel(self.flash_model)
            prompt = f"""
            Analyze the sentiment of the following news headlines for a trading asset.
            Respond strictly in valid JSON format with the following keys:
            - "signal": "BUY", "SELL", or "HOLD"
            - "confidence": a float between 0.0 and 1.0
            - "reasoning": a brief explanation of the sentiment
            
            Headlines:
            {json.dumps(headlines)}
            """
            
            response = model.generate_content(prompt)
            # Find JSON block
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"signal": "HOLD", "confidence": 0.5, "reasoning": f"Error: {e}"}
            
    def supervisor_debate(self, technician_signal: str, technician_rebuttal: str, critic_objections: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizes the debate into a final weighted confidence and reasoning."""
        if not self.api_key:
            return {"confidence": 0.5, "reasoning": "Debate bypassed due to missing API key."}
            
        try:
            model = genai.GenerativeModel(self.pro_model)
            prompt = f"""
            You are the Manager Agent of an AI trading committee. Read the following debate and synthesize a final decision.
            
            Market Context: {json.dumps(context)}
            Analyst Proposed Signal: {technician_signal}
            Risk Auditor Objections: {json.dumps(critic_objections)}
            Analyst Rebuttal: {technician_rebuttal}
            
            Task:
            Evaluate the validity of the technical setup against the critic's concerns.
            Output your final synthesized analysis in strictly valid JSON format:
            {{
                "confidence_adj": <float between -0.2 and 0.2 to adjust the base confidence>,
                "reasoning": "<your synthesized explanation>"
            }}
            """
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"Supervisor debate failed: {e}")
            return {"confidence_adj": 0.0, "reasoning": f"Debate synthesis failed: {e}"}
            
    def function_call_trade(self, debate_summary: str, expected_signal: str) -> Dict[str, Any]:
        """Optional function calling to trigger mock trade execution based on debate."""
        # Simplified implementation for now; could use GenAI tools feature
        pass
