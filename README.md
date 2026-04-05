# Quantryst: Multi-Agent Trading System

> A Quantitative Trading Research platform.

![Quantryst Dashboard Prototype](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB)
![ML](https://img.shields.io/badge/ML-XGBoost%20%2B%20LightGBM-F05340)

Quantryst is an advanced quantitative trading research platform that utilizes a collaborative multi-agent committee to generate high-probability market entries. Unlike traditional single-model systems, Quantryst utilizes a **Self-Correcting Adversarial Committee** designed to replicate institutional decision-making.ents, moderated by a centralized Manager, to ensure every signal is stress-tested before execution.

🌍 **Live Demo:**
-----

## 📊 Dashboard Preview

### Trading Signal Panel

**Signal OFF**
![Signal Off](https://github.com/user-attachments/assets/16b0b084-de57-4150-9d8f-64289f3f9982)

**Signal ON**
![Signal On](https://github.com/user-attachments/assets/0e0c2713-0a8c-41e4-bc20-1166164329c7)

### Agent Analysis
<img width="1650" height="478" alt="image" src="https://github.com/user-attachments/assets/5ce3ac2a-5a4e-48f7-82a3-b70c464c930c" />

### Backtesting Results
![Backtest](https://github.com/user-attachments/assets/08e88b0a-1d21-44ed-a98c-1fdd66c75e92)

### Multi Asset Overview
![Multi Asset](https://github.com/user-attachments/assets/e58fe26d-d5b7-4048-b73f-29147876e70d)


## 🏗️ Agent Committee System Architecture

The core of Quantryst is its **Adversarial Committee Architecture**, which mimics the workflow of a high-frequency hedge fund trading desk.

<img width="1602" height="613" alt="image" src="https://github.com/user-attachments/assets/be4f8046-1b57-4fd5-82ed-c0de2ee4c712" />


### 1. The Analyst (Technical & Order Flow)
Responsible for identifying structural market shifts and liquidity-driven opportunities.
*   **Market Structure Detection**: Identifies "Break of Structure" (BoS) and "Change of Character" (ChoCH) to determine the true price trend.
*   **Multi-Timeframe Confluence (MTA)**: Synchronizes short-term momentum with long-term trend alignment to find high-probability crossover points.
*   **Order Flow Analysis**: Utilizes Volume Weighted Average Price (VWAP) and Volume Ratios to identify institutional accumulation or distribution zones.
*   **Liquidity Gap Mapping**: Flags price imbalances and "Fair Value Gaps" (FVG) as potential magnets for price action.

### 2. The Sentiment Strategist (Macro & Narrative)
Monitors the global regime and identifies when the "Price Narrative" diverges from the "Market Facts."
*   **Market Regime Detection**: Classifies the current environment into "Risk-On" or "Risk-Off" states to adjust the committee's bias.
*   **Sentiment Divergence**: Detects instances where news headlines and social sentiment are at odds with price action, flagging potential exhaustion or contrarian opportunities.
*   **Narrative Strength Scoring**: Assigns a conviction score to the current market narrative based on macroeconomic catalyst decay.

### 3. The Risk Auditor (Adversarial Thesis)
Acting as the committee's "Devil's Advocate," this agent's sole purpose is to find reasons *not* to take a trade.
*   **Counter-Thesis Generation**: Explicitly builds the case for the opposite side of every proposed trade.
*   **Expected Value ($EV$) Calculation**: Uses real-time agent win-rates to determine if the statistical expected value of the trade justifies the risk.
*   **Half-Kelly Position Sizing**: Dynamically scales entry size based on the committee’s current reliability and ATR-derived volatility.
*   **ATR-Based Risk/Reward**: Automatically sets stop-loss and take-profit targets based on N-period Average True Range.

### 4. The Manager (Supervisor Agent)
The centralized intelligence that moderates the adversarial debate.
*   **Debate Synthesis**: Forces the committee into a "Deep Thinking" debate loop where agents must defend their signals against the Auditor's objections.
*   **Institutional Selective Filter**: Enforces the **Alpha Floor** (weighted edge ≥ 0.45), suppressing noisy signals into a "No-Trade Zone" to preserve capital.
*   **Consensus Generation**: Synthesizes conflicting signals into a final, high-conviction decision using a dynamically weighted consensus model.

---

## ⚖️ Dynamic Reliability Weighting

Quantryst implements a self-evolving weight system that rewards accuracy and punishes failures in real-time.

1.  Institutional Alpha Filter**: A strict "No-Trade Zone" floor. The committee must reach a weighted edge of **0.45** and a synthesized confidence of **0.65** to even *proposal* a trade. Anything less is automatically suppressed as a `HOLD`.
2.  Performance-Based Calibration**: Every agent's confidence is now anchored to their **15-minute EMA Win Rate**. 
    *   **Analyst (Technician)**: Gains influence when structural breakouts are confirmed.
    *   **Strategist (Narrator)**: Risk-haircuts are dynamically relaxed when sentiment reliability is high.
3.  High-Frequency Sync**: Dashboard and Agent "Live Thoughts" now refresh every **30 seconds** for real-time validation.

---

## 💻 Tech Stack

*   **Logic Engine**: Python (Advanced Feature Engineering, Signal Resolution)
*   **API Layer**: FastAPI (WebSocket streaming for live "Agent Thoughts")
*   **Machine Learning**: XGBoost & LightGBM (Baseline direction prediction)
*   **Frontend**: React (Vite) + Tailwind CSS (Custom dark-terminal UI)
*   **Data Pipeline**: Binance & Yahoo Finance (Real-time and historical OHLCV)

---

## 🚀 Installation and Setup

### 1. Prerequisites
*   Python 3.10+
*   Node.js 18+
*   Gemini API Key (Google AI Studio)

### 2. Backend Installation
```bash
# Clone and enter directory
git clone https://github.com/Daniswara369/ai-quant-trading-dashboard.git
cd ai-quant-trading-dashboard

# Switch to the institutional logic branch
git checkout v2-multi-agent-logic

# Install dependencies
pip install -r requirements.txt

# Configure Environment
# Create a .env file and add your GEMINI_API_KEY
echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env

# Start the API server
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

### 3. Frontend Installation
```bash
cd frontend
npm install
npm run dev
```
The dashboard will be available at `http://localhost:6001`.

---

## 📈 Dashboard Features

*   **Auto-Pilot Mode**: Enable the 15-minute execution loop to let the system continuously generate, track, and grade trade signals without manual intervention.
*   **Live Thoughts**: Watch the WebSocket stream of the agents' raw "thinking process" and adversarial debate as it happens.
*   **Dynamic Weight Visualization**: Monitor individual agent reliability percentages as they update based on markout performance.

---

## ⚠️ Disclaimer
This software is for research and educational purposes only. Automated trading involves significant risk of loss. The authors assume no responsibility for any financial losses incurred through the use of this software. Always test strategies thoroughly in a simulated environment before considering live application.
