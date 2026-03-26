# Quantryst AI Trading Analytics

> A full-stack Quantitative Trading Research platform powered by Machine Learning.

![Quantryst Dashboard Prototype](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB)
![ML](https://img.shields.io/badge/ML-XGBoost%20%2B%20LightGBM-F05340)

Quantryst is an end-to-end algorithmic trading system designed for quantitative research. It actively fetches market data, engineers complex technical indicators, trains machine learning models to predict price movements, simulates historical strategy performance (backtesting), and visualizes the results on a real-time reactive dashboard.

🌍 **Live Demo:** [https://ai-quant-trading-dashboard.vercel.app/]

Built initially as a terminal tool and evolved into a production-grade web application, this platform supports **Crypto (BTC, ETH), US Stocks (AAPL, MSFT, TSLA), and Forex pairs (EURUSD, GBPUSD, USDJPY)**.

## 📊 Dashboard Preview

### Trading Signal Panel

**Signal OFF**
![Signal Off](https://github.com/user-attachments/assets/16b0b084-de57-4150-9d8f-64289f3f9982)

**Signal ON**
![Signal On](https://github.com/user-attachments/assets/0e0c2713-0a8c-41e4-bc20-1166164329c7)

### Backtesting Results
![Backtest](https://github.com/user-attachments/assets/08e88b0a-1d21-44ed-a98c-1fdd66c75e92)

### Multi Asset Overview
![Multi Asset](https://github.com/user-attachments/assets/e58fe26d-d5b7-4048-b73f-29147876e70d)


## ✨ Key Features

- **Automated Data Pipelines**: Integrates with Binance, yfinance, and Yahoo Finance to fetch multi-asset OHLCV data.
- **Advanced Feature Engineering**: Computes 40+ quantitative indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, ADX) to feed the ML models.
- **Machine Learning Engine**: Trains XGBoost, Random Forest, LightGBM, and LSTM models to predict short-term price direction.
- **Robust Backtesting**: A custom event-driven backtesting engine that calculates Total Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, and generates Equity Curves.
- **Modern Dashboard**: A custom-built dark trading terminal (React + Tailwind CSS + Plotly) to visualize live signals and historical performance.

## 🛠️ Tech Stack

**Backend (Trading Engine & API)**
- **Python**: Core logic, data processing, and ML training.
- **FastAPI**: RESTful API layer serving predictions and backtest results.
- **Pandas & NumPy**: Heavy lifting for feature engineering and vector manipulation.
- **Scikit-learn / XGBoost**: Machine learning algorithms.

**Frontend (Dashboard)**
- **React (Vite)**: Component-based robust UI.
- **Tailwind CSS**: Custom styling and sleek animations.
- **Plotly.js**: Interactive financial charts (candlesticks, indicator overlays).
- **Axios**: API client management.

---

## 🚀 How to Run Locally

Because Quantryst is a decoupled client-server application, you need to run both the backend API and the frontend UI.

### 1. Start the API Server

First, install the Python dependencies and boot the FastAPI backend:

```bash
# Clone the repository
git clone https://github.com/Daniswara369/ai-quant-trading-dashboard.git
cd ai-quant-trading-dashboard

# Set up a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Start the API server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`. You can view the auto-generated Swagger documentation at `http://localhost:8000/docs`.*

### 2. Start the React Dashboard

In a new terminal window, navigate to the frontend directory:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

*The dashboard will be available at `http://localhost:6001`.*

---

## 🧠 Training New Models

The system comes with a few pre-trained XGBoost models for the default assets so the dashboard works out of the box. However, you can train new models or experiment with different algorithms easily via the CLI.

```bash
# Train an XGBoost model on Bitcoin with a 1-hour timeframe
python main.py --mode train --symbol BTCUSDT --timeframe 1h --model xgboost

# Run a dedicated backtest simulation on Apple stock
python main.py --mode backtest --symbol AAPL --market stock --timeframe 1d --model lightgbm

# Generate immediate buy/sell signals without the dashboard
python main.py --mode signal --symbol EURUSD --market forex --timeframe 15m --model random_forest
```

---

## 📅 Roadmap / Future Enhancements

- [ ] Connect a live exchange API (e.g., Binance, Alpaca) to execute actual paper trades securely.
- [ ] Add deep learning options (Transformers) alongside the current LSTM implementation.
- [ ] Implement a user authentication system for personal portfolio tracking.

## ⚠️ Disclaimer

**This project is for educational and research purposes only.** Do not use these models to trade real money without understanding the inherent risks of programmatic trading. The financial markets are highly volatile, and past performance calculated in backtests does not guarantee future results.
