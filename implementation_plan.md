# Quantryst: Multi-Agent Trading System Upgrade

Refactor the existing Quantryst ML dashboard into an autonomous, multi-agent trading system powered by Gemini. A **Committee of Agents** (Technician, Narrator, Critic, Supervisor) collaborates via an adversarial debate loop to validate signals, manage risk, and generate explainable trade reasoning.

## User Review Required

> [!IMPORTANT]
> **Gemini API Key**: A `GEMINI_API_KEY` environment variable is required. Uses `google-generativeai` free tier (Flash for sentiment, Pro for supervisor).

> [!WARNING]
> **Breaking Changes to Risk Manager**: MDD override drops from 20% → **5%**. Position sizing switches to **Half-Kelly** (0.5×f*) instead of full Kelly for safety given ~53% model accuracy.

> [!IMPORTANT]
> **Communication**: Uses **local WebSocket** for real-time agent thoughts (no Google Cloud Pub/Sub).

---

## Proposed Changes

### Agent Framework (`agents/`)

#### [NEW] [base_agent.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/agents/base_agent.py)
- ABC defining agent interface: `analyze(context) → AgentOutput`
- `AgentOutput`: `agent_name`, [signal](file:///e:/Project%20ML%20AI/agent-driven-trading-system/api/server.py#139-178), `confidence`, `reasoning`, `metadata`, `trigger_price`, `valid_until`
- `AgentContext`: `symbol`, `timeframe`, `ohlcv_df`, [indicators](file:///e:/Project%20ML%20AI/agent-driven-trading-system/features/technical_indicators.py#105-122), `timestamp`

#### [NEW] [technician_agent.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/agents/technician_agent.py)
- **Rules-based** (no LLM). Parses OHLCV + all indicators from [features/](file:///e:/Project%20ML%20AI/agent-driven-trading-system/features/quant_features.py#79-84)
- Detects: RSI divergence, MACD crossover, Bollinger squeeze, volume anomalies, golden/death crosses
- Also exposes `rebuttal(critic_objections) → str` for the debate loop — provides counter-arguments using secondary indicators (e.g. Volume Profile, ATR)

#### [NEW] [narrator_agent.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/agents/narrator_agent.py)
- Uses **Gemini 1.5 Flash** for news sentiment analysis
- Accepts headlines (initially from a configurable mock feed; extensible to GNews API)
- Returns structured JSON: `{signal, confidence, reasoning}`
- Falls back to HOLD/0.5 on API errors

#### [NEW] [critic_agent.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/agents/critic_agent.py)
- **Rules-based** "Skeptical" agent — explicitly finds reasons **not** to trade
- Detects: bull/bear traps, price-volume divergence, false breakouts, RSI extremes
- Exposes `challenge(technician_signal) → list[str]` — returns 3 specific objections for the debate loop

#### [NEW] [supervisor_agent.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/agents/supervisor_agent.py)

**Regime-Switching Dynamic Weights:**
First identifies the Market Regime, then adjusts weights:

| Regime | Technician | Narrator | Critic |
|---|---|---|---|
| **Trending** | 0.60 | 0.15 | 0.25 |
| **Ranging** | 0.20 | 0.30 | **0.50** |
| **News-Volatile** | 0.10 | **0.70** | 0.20 |
| **Default** | 0.40 | 0.25 | 0.35 |

**Adversarial "Red Team" Debate Loop:**
1. Technician proposes a signal (BUY/SELL)
2. Supervisor asks Critic: *"Find 3 reasons this signal is a trap"*
3. Supervisor asks Technician for a **rebuttal** using secondary indicators
4. **Gemini 1.5 Pro** synthesizes the debate → final weighted confidence + XAI reasoning
5. Uses Gemini **Function Calling** to optionally trigger `execute_mock_trade()`

---

### Core Shared Models (`core/`)

#### [NEW] [schemas.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/core/schemas.py)
- Pydantic models: `AgentOutput`, `AgentContext`, `TradeExecution`, `WeightedConsensus`, `MarketRegime`
- **Stale Signal Guard**: `AgentOutput` includes `trigger_price` and `valid_until` timestamp
- Server rejects execution if price deviates **>0.15%** from `trigger_price`

#### [NEW] [gemini_client.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/core/gemini_client.py)
- Wrapper around `google-generativeai` with rate limiting and retries
- Methods: `analyze_sentiment()`, `supervisor_debate()`, `function_call_trade()`
- Configurable model selection (Flash vs Pro)

#### [NEW] [__init__.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/core/__init__.py)

---

### Risk Management Upgrades

#### [MODIFY] [risk_manager.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/strategies/risk_manager.py)
- **5% MDD Hard Override**: New `hard_mdd_limit=0.05` — kills ALL signals when breached
- **Half-Kelly Position Sizing**: `kelly_position_size(win_prob, win_loss_ratio)` returns `0.5 × f*` for safety margin given imprecise probability estimates
- Existing [calculate_position_size](file:///e:/Project%20ML%20AI/agent-driven-trading-system/strategies/risk_manager.py#42-70) gains `use_kelly=True` parameter

#### [MODIFY] [config.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/config.py)
- Add `GEMINI_API_KEY`, `GEMINI_FLASH_MODEL`, `GEMINI_PRO_MODEL`
- Add `AGENT_WEIGHTS` (default regime weights)
- Add `MDD_HARD_OVERRIDE = 0.05`, `TAKER_FEE = 0.001`
- Add `STALE_SIGNAL_THRESHOLD = 0.0015` (0.15%)
- Add `KELLY_FRACTION = 0.5` (Half-Kelly)

---

### Backend Architecture Upgrade

#### [MODIFY] [server.py](file:///e:/Project%20ML%20AI/agent-driven-trading-system/api/server.py)
- **WebSocket** `/ws/agent-thoughts` — streams real-time agent reasoning to frontend
- `POST /api/agent-signal` — runs full agent pipeline (debate loop) → returns weighted consensus
- `POST /api/mock-trade` — Gemini function-call mock trade execution with **stale signal guard**
- `GET /api/agent-status` — last pipeline result

---

### Frontend "Agent Thoughts" UI

#### [NEW] [AgentThoughts.jsx](file:///e:/Project%20ML%20AI/agent-driven-trading-system/frontend/src/components/AgentThoughts.jsx)
- Real-time panel showing each agent's output: signal, confidence bar, reasoning
- WebSocket connection to `/ws/agent-thoughts`
- Agent-specific icons (🔧 Technician, 📰 Narrator, 🧐 Critic, 👔 Supervisor)
- Shows: debate transcript, weighted confidence gauge, regime indicator

#### [MODIFY] [App.jsx](file:///e:/Project%20ML%20AI/agent-driven-trading-system/frontend/src/App.jsx)
- Add `AgentThoughts` component + "Run Agent Analysis" button

#### [MODIFY] [index.css](file:///e:/Project%20ML%20AI/agent-driven-trading-system/frontend/src/index.css)
- Agent thought card styles, confidence gauge, debate animation

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///e:/Project%20ML%20AI/agent-driven-trading-system/requirements.txt)
- Add `google-generativeai>=0.3.0`, `websockets>=12.0`, `pydantic>=2.0`

---

## Verification Plan

### Automated Tests
1. **Agent imports** — verify all 4 agents instantiate and produce valid `AgentOutput`
2. **Risk Manager** — verify Half-Kelly sizing and 5% MDD hard override
3. **Stale Signal Guard** — verify trade rejection when price deviates >0.15%
4. **FastAPI endpoints** — `curl` test new `/api/agent-signal` and `/api/agent-status`

### Manual Verification
1. Start backend → frontend → open browser
2. Click "Run Agent Analysis" for BTCUSDT
3. Verify Agent Thoughts panel shows 4 agents + debate transcript
4. Verify weighted confidence gauge w/ regime indicator
5. Check WebSocket in DevTools Network tab
