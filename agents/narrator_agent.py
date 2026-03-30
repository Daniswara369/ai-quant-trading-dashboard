import logging
import os
import re
import json
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple
from agents.base_agent import BaseAgent
from core.schemas import AgentOutput, AgentContext
from core.gemini_client import GeminiClient
import requests

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

    # -----------------------------
    # Local (free) intelligence
    # -----------------------------
    def _infer_market_type(self, symbol: str) -> str:
        """Best-effort market type inference (crypto/stock/forex)."""
        try:
            from data.data_pipeline import detect_market_type
            return detect_market_type(symbol)
        except Exception:
            s = (symbol or "").upper()
            if s.endswith("USDT") or s.endswith("BTC"):
                return "crypto"
            if len(s) == 6 and s[:3].isalpha() and s[3:].isalpha():
                return "forex"
            return "stock"

    def _vader_analyze(self, headlines: List[str]) -> Dict[str, Any]:
        """
        Free local sentiment scoring using NLTK VADER.
        Returns a dict with mean/variance and per-headline scores.
        """
        cleaned = [h.strip() for h in (headlines or []) if isinstance(h, str) and h.strip()]
        if not cleaned:
            return {"available": False, "mean": 0.0, "variance": 0.0, "stdev": 0.0, "scores": []}

        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
            import nltk  # type: ignore
            try:
                _ = SentimentIntensityAnalyzer()
            except Exception:
                nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
            compounds = [float(sia.polarity_scores(h).get("compound", 0.0)) for h in cleaned]
        except Exception as e:
            logger.warning(f"VADER unavailable; falling back to neutral. err={e}")
            return {"available": False, "mean": 0.0, "variance": 0.0, "stdev": 0.0, "scores": []}

        mean = float(statistics.mean(compounds)) if compounds else 0.0
        variance = float(statistics.pvariance(compounds)) if len(compounds) >= 2 else 0.0
        stdev = float(math.sqrt(variance)) if variance > 0 else 0.0
        return {
            "available": True,
            "mean": mean,
            "variance": variance,
            "stdev": stdev,
            "scores": [{"headline": h, "compound": c} for h, c in zip(cleaned, compounds)],
        }

    def _headline_features(self, headlines: List[str], market_type: str) -> Dict[str, Any]:
        """Cheap keyword-based macro features (geopolitics, FUD, central bank tone, etc.)."""
        text = " | ".join([h.lower() for h in (headlines or []) if isinstance(h, str)])
        def _count(words: List[str]) -> int:
            return sum(1 for w in words if w in text)

        geopolitical = _count(["war", "conflict", "sanction", "tariff", "missile", "election", "coup", "trade war", "china", "taiwan", "iran", "israel", "russia", "ukraine", "nato"])
        energy = _count(["oil", "opec", "gas", "energy crisis", "pipeline", "brent", "wti"])
        risk_off = _count(["recession", "default", "bank run", "credit event", "liquidity", "panic", "risk-off", "downgrade"])

        if market_type == "crypto":
            etf_flows = _count(["etf", "inflow", "outflow", "spot etf", "blackrock", "fidelity", "ark", "grayscale"])
            fud = _count(["fud", "hack", "exploit", "rug", "scam", "sec", "lawsuit", "ban", "delist", "liquidation"])
            return {"geopolitical_hits": geopolitical, "energy_hits": energy, "risk_off_hits": risk_off, "etf_flow_hits": etf_flows, "fud_hits": fud}

        if market_type == "forex":
            cb = _count(["fed", "ecb", "boe", "boj", "pbo", "rba", "boc", "rate", "rates", "hike", "cut", "minutes", "speech", "inflation", "cpi", "ppi", "jobs", "nfp", "gdp", "yield"])
            return {"geopolitical_hits": geopolitical, "energy_hits": energy, "risk_off_hits": risk_off, "central_bank_hits": cb}

        # stock
        vix = _count(["vix", "volatility index", "fear gauge"])
        sector = _count(["sector", "tech", "financials", "energy", "healthcare", "semiconductor", "banks", "utilities", "consumer", "industrial"])
        return {"geopolitical_hits": geopolitical, "energy_hits": energy, "risk_off_hits": risk_off, "vix_hits": vix, "sector_hits": sector}

    def _macro_harvester(self) -> List[str]:
        """
        MacroHarvester fallback if upstream didn't provide global headlines.
        Uses a free public endpoint (GDELT DOC) to grab top 3 macro stories,
        focusing on Hormuz / Fed / OPEC themes.
        """
        query = "Hormuz OR Fed OR OPEC"
        try:
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 3,
                "sort": "HybridRel",
            }
            resp = requests.get(url, params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                arts = data.get("articles") or []
                titles = []
                for a in arts[:3]:
                    t = (a.get("title") or "").strip()
                    if t:
                        titles.append(t)
                if titles:
                    return titles
        except Exception as e:
            logger.warning(f"MacroHarvester failed: {e}")

        return [
            "Fed officials reiterate data-dependent stance as inflation risks persist.",
            "OPEC+ signals continued supply discipline amid volatile energy markets.",
            "Strait of Hormuz remains a key energy chokepoint; risk premium stays elevated.",
        ]

    def _determine_global_regime(self, global_headlines: List[str]) -> Dict[str, Any]:
        """
        Global regime filter (macro-first). This is deliberately rules-based and tail-risk aware.
        Returns {regime, tail_risk, drivers[]}.
        """
        text = " | ".join([h.lower() for h in (global_headlines or []) if isinstance(h, str)])
        drivers: List[str] = []

        hormuz = any(k in text for k in ["hormuz", "strait of hormuz"])
        opec = "opec" in text or "opec+" in text
        energy = any(k in text for k in ["oil", "brent", "wti", "gas", "energy"]) or opec or hormuz
        fed = any(k in text for k in ["fed", "fomc", "powell"])
        hawkish = any(k in text for k in ["hawkish", "hike", "higher for longer", "tightening", "sticky inflation"])
        dovish = any(k in text for k in ["dovish", "cut", "easing", "disinflation"])
        war = any(k in text for k in ["war", "conflict", "strike", "missile", "sanction", "escalat", "attack"])

        if hormuz:
            drivers.append("Hormuz energy chokepoint risk (impacts ~20% of global oil flows; tail-risk premium).")
        if opec:
            drivers.append("OPEC supply policy / discipline (oil price sensitivity).")
        if fed:
            drivers.append("Fed policy/rhetoric (global discount-rate driver).")
        if war:
            drivers.append("Geopolitical escalation risk (risk-off impulse).")

        # World model (2026): energy shock -> inflation -> hawkish Fed -> pressure on risk assets
        if energy and (hormuz or war):
            return {"regime": "ENERGY_SHOCK_RISK_OFF", "tail_risk": True, "drivers": drivers or ["Energy/geopolitical risk premium elevated."]}
        if fed and hawkish:
            return {"regime": "HAWKISH_FED_RISK_OFF", "tail_risk": False, "drivers": drivers or ["Hawkish Fed pressures risk assets."]}
        if fed and dovish:
            return {"regime": "DOVISH_FED_RISK_ON", "tail_risk": False, "drivers": drivers or ["Dovish pivot supports risk assets."]}
        if war or (energy and opec):
            return {"regime": "GEOPOLITICAL_RISK_OFF", "tail_risk": bool(hormuz), "drivers": drivers or ["Geopolitical/energy uncertainty biases risk-off."]}

        return {"regime": "GLOBAL_NEUTRAL", "tail_risk": False, "drivers": drivers or ["No dominant macro shock detected."]}

    def _price_change_24h(self, context: AgentContext) -> Optional[float]:
        """Compute 24h % change using available OHLCV and timeframe."""
        df = context.ohlcv_df
        if df is None or getattr(df, "empty", True):
            return None
        try:
            closes = df["Close"] if "Close" in df.columns else None
            if closes is None or len(closes) < 2:
                return None
            tf = (context.timeframe or "1h").lower().strip()
            # bars per 24h
            bars = None
            m = re.fullmatch(r"(\d+)(m|h|d)", tf)
            if m:
                n = int(m.group(1))
                unit = m.group(2)
                if unit == "m":
                    bars = int(round(24 * 60 / max(n, 1)))
                elif unit == "h":
                    bars = int(round(24 / max(n, 1)))
                elif unit == "d":
                    bars = 1
            if bars is None:
                bars = 24  # conservative default
            if len(closes) <= bars:
                bars = max(min(len(closes) - 1, bars), 1)
            now = float(closes.iloc[-1])
            prev = float(closes.iloc[-1 - bars])
            if prev == 0:
                return None
            return (now - prev) / prev
        except Exception:
            return None

    def _detect_sentiment_divergence(self, local_sentiment_mean: float, price_change_24h: Optional[float]) -> Dict[str, Any]:
        """
        Divergence = headlines bullish but price falling, or vice versa.
        Uses local sentiment (VADER mean) and 24h price change.
        """
        divergence = {"detected": False, "type": None, "detail": ""}
        if price_change_24h is None:
            return divergence

        # sentiment bias
        headline_bias = "BULLISH" if local_sentiment_mean > 0.15 else ("BEARISH" if local_sentiment_mean < -0.15 else "NEUTRAL")
        price_bias = "BULLISH" if price_change_24h > 0.01 else ("BEARISH" if price_change_24h < -0.01 else "NEUTRAL")

        if headline_bias == "BULLISH" and price_bias == "BEARISH":
            divergence["detected"] = True
            divergence["type"] = "BEARISH_DIVERGENCE"
            divergence["detail"] = "Headlines are net-bullish but 24h price momentum is negative; risk of distribution or narrative trap."
        elif headline_bias == "BEARISH" and price_bias == "BULLISH":
            divergence["detected"] = True
            divergence["type"] = "BULLISH_DIVERGENCE"
            divergence["detail"] = "Headlines are net-bearish but 24h price momentum is positive; risk of contrarian squeeze or accumulation."

        return divergence

    # -----------------------------
    # LLM calls + failover
    # -----------------------------
    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """Extract first JSON object from model output."""
        if not text:
            raise ValueError("Empty model response.")
        t = text.strip()
        if t.startswith("```json"):
            t = t[7:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        # If it's pure JSON, parse directly; otherwise extract first {...}
        try:
            return json.loads(t)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            raise ValueError(f"Could not locate JSON in response: {t[:180]}")
        return json.loads(m.group(0))

    def _call_gemini_json(self, prompt: str) -> Tuple[Dict[str, Any], str]:
        """Call Gemini and parse JSON. Returns (result, provider)."""
        import google.generativeai as genai  # local import to avoid hard crash if missing
        model = genai.GenerativeModel(self.gemini_client.flash_model)
        response = model.generate_content(prompt)
        return self._extract_json_object(getattr(response, "text", "") or ""), "gemini"

    def _call_groq_json(self, prompt: str) -> Tuple[Dict[str, Any], str]:
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
        resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=20)
        if resp.status_code >= 400:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:180]}")
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        return self._extract_json_object(content), "groq:llama-3.3-70b-versatile"

    def analyze(self, context: AgentContext) -> AgentOutput:
        """
        Institutional-grade macro & sentiment analysis:
        1. Risk-On / Risk-Off regime detection (via Gemini)
        2. Sentiment Divergence (headlines vs price momentum)
        3. Narrative Strength rating
        """
        headlines = context.metadata.get("news_headlines", [])
        global_headlines = context.metadata.get("global_headlines", [])
        if not global_headlines:
            global_headlines = self._macro_harvester()

        # Get own reliability weight from context
        my_weight = context.metadata.get("agent_weights", {}).get(self.name, None)
        weight_note = ""
        if my_weight is not None:
            weight_note = f"Given my current reliability of {my_weight:.2f}, I apply proportionally higher skepticism to marginal/late-cycle signals and demand stronger evidence before issuing BUY/SELL."

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
                reasoning=(regime_note + (f" {weight_note}" if weight_note else "")).strip(),
                metadata={
                    "regime": "AMBIGUOUS",
                    "narrative_strength": 0,
                    "divergence": {"detected": False, "type": None, "detail": ""},
                    "api_provider": "local:no_news",
                    "global_regime": self._determine_global_regime(global_headlines),
                    "global_headlines_count": len(global_headlines),
                },
            )

        symbol = context.symbol
        market_type = self._infer_market_type(symbol)

        # Global filter first (macro regime + tail-risk)
        global_regime = self._determine_global_regime(global_headlines)
        global_vader = self._vader_analyze(global_headlines)
        global_mean = float(global_vader.get("mean", 0.0) or 0.0)

        # Local VADER significance + entropy (variance) gate
        vader = self._vader_analyze(headlines)
        local_mean = float(vader.get("mean", 0.0) or 0.0)
        local_stdev = float(vader.get("stdev", 0.0) or 0.0)
        # Entropy proxy: high stdev => contradictory headlines
        entropy_score = 0.0
        if vader.get("available") and local_stdev > 0:
            # scale stdev (~0.0-0.6 typical) into 0..1
            entropy_score = float(max(0.0, min(local_stdev / 0.35, 1.0)))

        price_change_24h = self._price_change_24h(context)
        divergence = self._detect_sentiment_divergence(local_mean, price_change_24h)

        # Macro-vs-micro narrative divergence: macro bearish while symbol news bullish (or vice versa)
        narrative_divergence = {"detected": False, "type": None, "detail": ""}
        try:
            macro_bearish = global_regime.get("regime", "").endswith("RISK_OFF") or global_mean < -0.15
            macro_bullish = global_regime.get("regime", "") == "DOVISH_FED_RISK_ON" or global_mean > 0.15
            micro_bullish = local_mean > 0.15
            micro_bearish = local_mean < -0.15
            if macro_bearish and micro_bullish:
                narrative_divergence = {
                    "detected": True,
                    "type": "MACRO_BEARISH_MICRO_BULLISH",
                    "detail": "Macro backdrop is risk-off while symbol headlines skew bullish; treat as fragile/tail-risked rally.",
                }
            elif macro_bullish and micro_bearish:
                narrative_divergence = {
                    "detected": True,
                    "type": "MACRO_BULLISH_MICRO_BEARISH",
                    "detail": "Macro backdrop is supportive/risk-on while symbol headlines skew bearish; potential idiosyncratic risk.",
                }
        except Exception:
            pass

        # If sentiment is market-noise locally AND macro is not screaming, HOLD to save API quota
        if vader.get("available") and (-0.1 < local_mean < 0.1) and not narrative_divergence.get("detected") and not global_regime.get("tail_risk"):
            reasoning = (
                "Thesis: Micro (symbol) headlines are statistically near-neutral (noise), and the macro tape is not presenting a dominant shock.\n"
                "Counter-thesis: Macro can flip quickly in 2026 (energy chokepoints, Fed repricing), but that risk is not concentrated enough here to justify paid inference.\n"
                "Synthesis: HOLD locally; wait for clearer micro narrative, or macro confirmation."
            )
            if weight_note:
                reasoning += f"\n\n{weight_note}"
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.55 if my_weight is not None and my_weight >= 0.5 else 0.5,
                reasoning=reasoning.strip(),
                metadata={
                    "headlines_count": len(headlines),
                    "regime": "AMBIGUOUS",
                    "narrative_strength": 2,
                    "divergence": divergence,
                    "api_provider": "local:vader_gate",
                    "market_type": market_type,
                    "global_headlines_count": len(global_headlines),
                    "global_regime": global_regime,
                    "narrative_divergence": narrative_divergence,
                    "local_sentiment": {"mean": round(local_mean, 4), "stdev": round(local_stdev, 4), "entropy": round(entropy_score, 4)},
                    "global_sentiment": {"mean": round(global_mean, 4)},
                },
            )

        # Build the institutional-grade prompt for Gemini
        priced_in = {"flag": False, "detail": ""}
        if price_change_24h is not None:
            if local_mean >= 0.6 and price_change_24h > 0.05:
                priced_in = {"flag": True, "detail": "Bullish sentiment is extreme while price is already up >5% over 24h; upside may be partially priced in."}
            elif local_mean <= -0.6 and price_change_24h < -0.05:
                priced_in = {"flag": True, "detail": "Bearish sentiment is extreme while price is already down >5% over 24h; downside may be partially priced in."}

        feat = self._headline_features(headlines, market_type)
        divergence_context = f"SENTIMENT/PRICE DIVERGENCE: {divergence['detail']}" if divergence.get("detected") else "SENTIMENT/PRICE DIVERGENCE: None detected."
        priced_in_context = f"PRICED-IN CHECK: {priced_in['detail']}" if priced_in.get("flag") else "PRICED-IN CHECK: Not flagged."
        entropy_context = "NARRATIVE ENTROPY: Headlines are contradictory/high-entropy; bias toward HOLD unless evidence is overwhelming." if entropy_score >= 0.6 else "NARRATIVE ENTROPY: Acceptable alignment."

        try:
            prompt = f"""You are an institutional macro strategist at a top-tier hedge fund.

            2026 WORLD MODEL (pre-load):
            - The Strait of Hormuz is a major energy chokepoint affecting ~20% of global oil flows.
            - Energy shock -> inflation risk -> hawkish Fed reaction -> pressure on risk assets.
            - Tail-risk matters: a BUY during Hormuz escalation must be smaller / lower confidence.
            
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
            
            5. **Reasoning**: Institutional-grade. MUST follow Thesis -> Counter-thesis -> Synthesis.
               - Explicitly state how the current Reliability Weight (w) influenced your skepticism.
            
            CONTEXT:
            - Symbol: {symbol}
            - Market type: {market_type}
            - Global headlines (macro tape): {json.dumps(global_headlines)}
            - Global regime (rules-based): {json.dumps(global_regime)}
            - Local VADER sentiment mean: {local_mean:.3f} (stdev {local_stdev:.3f})
            - Global VADER sentiment mean: {global_mean:.3f}
            - 24h price change: {price_change_24h if price_change_24h is not None else "N/A"}
            - {divergence_context}
            - {priced_in_context}
            - {entropy_context}
            - Macro vs Micro divergence: {json.dumps(narrative_divergence)}
            - Instrument-aware cues: {json.dumps(feat)}
            - Reliability weight (w): {my_weight if my_weight is not None else "N/A"}

            INSTRUMENT RULES (hedge-fund heuristics):
            - Crypto: ETF flows amplify risk-on; regulation/hacks amplify risk-off; high tail-risk => discount confidence.
            - Stocks: USD/real rates up + energy shock => risk-off, especially for long-duration Tech; defensives outperform.
            - Forex: USD tends to be safe-haven in risk-off; rate differentials matter; central bank rhetoric dominates.
            
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

            # Primary: Gemini, Failover: Groq on 429 rate limit
            api_provider = "gemini"
            try:
                result, api_provider = self._call_gemini_json(prompt)
            except Exception as e:
                if ("429" in str(e)) or ("RESOURCE_EXHAUSTED" in str(e)) or ("quota" in str(e).lower()):
                    try:
                        result, api_provider = self._call_groq_json(prompt)
                    except Exception as e2:
                        raise RuntimeError(f"Rate-limited on Gemini and Groq failover failed: {e2}") from e2
                else:
                    raise

            # Adjust confidence if divergence detected
            signal = result.get("signal", "HOLD")
            confidence = float(result.get("confidence", 0.5))
            regime = result.get("regime", "AMBIGUOUS")
            narrative_strength = int(result.get("narrative_strength", 5))

            reasoning = (result.get("reasoning", "") or "").strip()
            if not reasoning:
                reasoning = "Thesis: Mixed or insufficient signal from headlines.\nCounter-thesis: Some catalysts may be underweighted by this sample.\nSynthesis: HOLD pending clearer narrative alignment."

            # ── Performance-Based Calibration ──
            # Fix 'underconfidence' or 'overconfidence' using ema_win_15m
            calibration_note = ""
            perf = context.metadata.get("agent_performance", {}).get(self.name, {})
            win_rate = perf.get("ema_win_15m", 0.5)
            n_resolved = perf.get("n_resolved", 0)
            
            # Calibration multiplier: if win_rate is strong, we 'relax' the gravity of the upcoming haircuts
            penalty_relax = 1.0
            if n_resolved >= 5:
                if win_rate > 0.52:
                    penalty_relax = 0.6 # Reduce the penalty magnitude by 40% if performing well
                    calibration_note = f" [Calibration: Risk-haircuts relaxed due to {win_rate:.1%} historical reliability]."
                elif win_rate < 0.48:
                    penalty_relax = 1.3 # Increase penalty weight if performing poorly
                    calibration_note = f" [Calibration: Confidence deflated due to {win_rate:.1%} historical reliability]."
                else:
                    calibration_note = f" [Calibration: Performance-anchored at {win_rate:.1%} win-rate]."

            # Hard risk controls: entropy and priced-in override "vibes"
            if entropy_score >= 0.75:
                signal = "HOLD"
                confidence = min(confidence, 0.55)
                reasoning += "\n\nEntropy override: Headlines are highly contradictory, so expected edge is low; biasing to HOLD."
            elif entropy_score >= 0.6:
                confidence = min(confidence, 0.65)
                reasoning += "\n\nEntropy penalty: Contradictory headlines reduce signal clarity; confidence is discounted."

            # --- Apply Haircuts (now penalty_relaxed) ---
            if divergence.get("detected"):
                penalty = 0.12 * penalty_relax
                confidence = max(min(confidence - penalty, confidence), 0.3)
                reasoning += f"\n\nDivergence noted: {divergence.get('detail')}"

            if narrative_divergence.get("detected"):
                penalty = 0.12 * penalty_relax
                confidence = max(confidence - penalty, 0.25)
                reasoning += f"\n\nMacro-vs-micro divergence: {narrative_divergence.get('detail')}"

            # Execution-aware tail-risk haircut (Hormuz / energy shock)
            if global_regime.get("tail_risk"):
                penalty = 0.10 * penalty_relax
                confidence = max(confidence - penalty, 0.2)
                reasoning += "\n\nTail-risk haircut: Hormuz/energy chokepoint risk elevates gap risk; sizing should be smaller / stops wider."

            if priced_in.get("flag"):
                penalty = 0.10 * penalty_relax
                confidence = max(confidence - penalty, 0.25)
                reasoning += f"\n\nPriced-in penalty: {priced_in.get('detail')}"

            if weight_note and (str(my_weight) not in reasoning):
                reasoning += f"\n\n{weight_note}"
            
            if calibration_note:
                reasoning += f"\n\n{calibration_note}"

            return AgentOutput(
                agent_name=self.name,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning.strip(),
                metadata={
                    "headlines_count": len(headlines),
                    "regime": regime,
                    "narrative_strength": narrative_strength,
                    "divergence": divergence,
                    "api_provider": api_provider,
                    "market_type": market_type,
                    "priced_in": priced_in,
                    "global_headlines_count": len(global_headlines),
                    "global_regime": global_regime,
                    "narrative_divergence": narrative_divergence,
                    "local_sentiment": {"mean": round(local_mean, 4), "stdev": round(local_stdev, 4), "entropy": round(entropy_score, 4)},
                    "global_sentiment": {"mean": round(global_mean, 4)},
                    "price_change_24h": price_change_24h,
                    "macro_features": feat,
                },
            )

        except Exception as e:
            logger.error(f"Sentiment Strategist failed: {e}")
            return AgentOutput(
                agent_name=self.name,
                signal="HOLD",
                confidence=0.5,
                reasoning=f"Sentiment API error: {e}",
                metadata={
                    "regime": "AMBIGUOUS",
                    "narrative_strength": 0,
                    "divergence": {"detected": False, "type": None, "detail": ""},
                    "api_provider": "error",
                    "global_regime": self._determine_global_regime(global_headlines),
                    "global_headlines_count": len(global_headlines),
                },
            )
