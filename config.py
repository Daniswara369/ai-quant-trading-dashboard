"""
Central configuration for the AI Quant Trading System.
"""
import os

# Manually load .env if it exists so we don't need external dependencies like python-dotenv
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# ─────────────────────────────────────────────
# API Keys (set via environment variables or fill in directly)
# ─────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-flash")

# ─────────────────────────────────────────────
# Agent Framework
# ─────────────────────────────────────────────
AGENT_WEIGHTS = {
    "Trending": {"Analyst": 0.60, "Sentiment Strategist": 0.15, "Risk Auditor": 0.25},
    "Ranging": {"Analyst": 0.20, "Sentiment Strategist": 0.30, "Risk Auditor": 0.50},
    "News-Volatile": {"Analyst": 0.10, "Sentiment Strategist": 0.70, "Risk Auditor": 0.20},
    "Default": {"Analyst": 0.40, "Sentiment Strategist": 0.25, "Risk Auditor": 0.35}
}
STALE_SIGNAL_THRESHOLD = 0.0015  # 0.15% deviation from trigger price


# ─────────────────────────────────────────────
# Asset Lists
# ─────────────────────────────────────────────
CRYPTO_ASSETS = ["BTCUSDT", "ETHUSDT"]
STOCK_ASSETS = ["AAPL", "MSFT", "TSLA"]
FOREX_ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

ALL_ASSETS = {
    "crypto": CRYPTO_ASSETS,
    "stock": STOCK_ASSETS,
    "forex": FOREX_ASSETS,
}

# Yahoo Finance ticker mapping for forex
FOREX_YAHOO_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
}

# ─────────────────────────────────────────────
# Timeframes
# ─────────────────────────────────────────────
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]

# Binance interval mapping
BINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
}

# yfinance interval mapping
YFINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
}

# yfinance period mapping (max data for each interval)
YFINANCE_PERIOD_MAP = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "1d": "max",
}

# ─────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories
for d in [DATA_CACHE_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# Trading Parameters
# ─────────────────────────────────────────────
SIGNAL_BUY_THRESHOLD = 0.6
SIGNAL_SELL_THRESHOLD = 0.6
INITIAL_CAPITAL = 100_000.0

# ─────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────
RISK_PER_TRADE = 0.01          # 1% of capital per trade
MAX_DRAWDOWN_LIMIT = 0.20      # 20% max drawdown before halting
MDD_HARD_OVERRIDE = 0.05       # 5% hard limit, skips all signals
MAX_PORTFOLIO_EXPOSURE = 0.50  # 50% max of capital in open positions
DEFAULT_STOP_LOSS_PCT = 0.02   # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.04 # 4% take profit
KELLY_FRACTION = 0.5           # Half-Kelly for sizing margin of safety
TAKER_FEE = 0.001              # 0.1% fee

# Spread simulation (in price fraction)
SPREAD_MAP = {
    "crypto": 0.001,   # 0.1%
    "stock": 0.0005,   # 0.05%
    "forex": 0.0002,   # 2 pips approx
}
TRANSACTION_COST = 0.001  # 0.1% per trade

# ─────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
}

LIGHTGBM_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 50, 70],
}

LSTM_PARAMS = {
    "units": 64,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "sequence_length": 30,
}

# ─────────────────────────────────────────────
# Real-Time Engine
# ─────────────────────────────────────────────
REALTIME_REFRESH_SECONDS = 300  # 5 minutes

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
SMA_PERIODS = [10, 20, 50]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCHASTIC_PERIOD = 14
VOLUME_MA_PERIOD = 20
ROLLING_VOL_WINDOW = 20
MOMENTUM_WINDOW = 10
