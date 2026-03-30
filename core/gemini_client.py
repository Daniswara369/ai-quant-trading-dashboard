import os
import json
import logging
import re
import time
import google.generativeai as genai
import requests
from typing import Dict, Any, List
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL

logger = logging.getLogger(__name__)


def _call_groq_json(prompt: str) -> Dict[str, Any]:
    """
    Groq OpenAI-compatible endpoint fallback.
    Requires GROQ_API_KEY in environment.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set for Groq failover.")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Return strictly valid JSON only. No markdown, no extra text."},
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:180]}")
    data = resp.json()
    content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    # Parse JSON from Groq response
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError(f"Could not locate JSON in Groq response: {text[:180]}")
        return json.loads(m.group(0))


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
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    # Find JSON block
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                        
                    return json.loads(text.strip())
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit in sentiment analysis. Retrying in {2 ** attempt}s...")
                        time.sleep(2 ** attempt)
                        continue
                    raise e
                    
        except Exception as e:
            # Failover to Groq on rate limit
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                logger.warning(f"Gemini rate-limited for sentiment. Failing over to Groq...")
                try:
                    return _call_groq_json(prompt)
                except Exception as e2:
                    logger.error(f"Groq failover also failed for sentiment: {e2}")
            logger.error(f"Sentiment analysis failed: {e}")
            return {"signal": "HOLD", "confidence": 0.5, "reasoning": f"Error: {e}"}
            
    def supervisor_debate(self, technician_signal: str, technician_rebuttal: str, critic_objections: List[str], context: Dict[str, Any], committee_snapshot: str = "") -> Dict[str, Any]:
        """Synthesizes the debate into a final weighted confidence and reasoning."""
        if not self.api_key:
            return {"confidence": 0.5, "reasoning": "Debate bypassed due to missing API key."}
            
        try:
            model = genai.GenerativeModel(self.pro_model)
            
            # Build the weight-aware prompt section
            weight_instruction = ""
            if committee_snapshot:
                weight_instruction = f"""
            
            === COMMITTEE RELIABILITY WEIGHTS (from historical performance tracking) ===
            {committee_snapshot}
            
            CRITICAL INSTRUCTION: You MUST weigh each agent's signal according to their current
            Reliability Weight shown above. An agent with a weight of 0.43 is significantly more
            trustworthy than one with 0.20. Factor these weights heavily into your final synthesis.
            If an agent with a high reliability weight disagrees with a low-weight agent, strongly
            favor the high-weight agent's position.
            """
            
            prompt = f"""
            You are the Manager Agent of an AI trading committee. Read the following debate and synthesize a final decision.
            
            Market Context: {json.dumps(context)}
            Analyst Proposed Signal: {technician_signal}
            Risk Auditor Objections: {json.dumps(critic_objections)}
            Analyst Rebuttal: {technician_rebuttal}
            {weight_instruction}
            Task:
            Evaluate the validity of the technical setup against the critic's concerns.
            Output your final synthesized analysis in strictly valid JSON format:
            {{
                "confidence_adj": <float between -0.2 and 0.2 to adjust the base confidence>,
                "reasoning": "<your synthesized explanation>"
            }}
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    
                    return json.loads(text.strip())
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit in supervisor debate. Retrying in {2 ** attempt}s...")
                        time.sleep(2 ** attempt)
                        continue
                    raise e
                    
        except Exception as e:
            # Failover to Groq on rate limit
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                logger.warning(f"Gemini rate-limited for supervisor debate. Failing over to Groq...")
                try:
                    result = _call_groq_json(prompt)
                    logger.info("Supervisor debate completed via Groq failover.")
                    return result
                except Exception as e2:
                    logger.error(f"Groq failover also failed for supervisor debate: {e2}")
            logger.error(f"Supervisor debate failed: {e}")
            return {"confidence_adj": 0.0, "reasoning": f"Debate synthesis failed: {e}"}
            
    def function_call_trade(self, debate_summary: str, expected_signal: str) -> Dict[str, Any]:
        """Optional function calling to trigger mock trade execution based on debate."""
        # Simplified implementation for now; could use GenAI tools feature
        pass
