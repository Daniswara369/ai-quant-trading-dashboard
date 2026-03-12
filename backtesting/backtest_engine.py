"""
Backtesting engine — simulate trading strategies on historical data.
"""
import pandas as pd
import numpy as np
import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, SPREAD_MAP,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
)
from strategies.risk_manager import RiskManager


class BacktestEngine:
    """
    Full-featured backtesting engine with position management,
    stop loss / take profit, and performance metrics.
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        transaction_cost: float = None,
        market_type: str = "crypto",
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ):
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.transaction_cost = transaction_cost or TRANSACTION_COST
        self.market_type = market_type
        self.spread = SPREAD_MAP.get(market_type, 0.001)
        self.stop_loss_pct = stop_loss_pct or DEFAULT_STOP_LOSS_PCT
        self.take_profit_pct = take_profit_pct or DEFAULT_TAKE_PROFIT_PCT
        
        # State
        self.capital = self.initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_direction = None
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        
        # Tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.risk_manager = RiskManager(self.initial_capital)
    
    def _execute_buy(self, price: float, timestamp, position_size: float = None):
        """Open a long position."""
        effective_price = price * (1 + self.spread / 2)
        
        if position_size is None:
            stop_loss = effective_price * (1 - self.stop_loss_pct)
            position_size = self.risk_manager.calculate_position_size(effective_price, stop_loss)
        
        cost = position_size * effective_price
        tx_cost = cost * self.transaction_cost
        
        if cost + tx_cost > self.capital:
            position_size = (self.capital * 0.95) / (effective_price * (1 + self.transaction_cost))
            cost = position_size * effective_price
            tx_cost = cost * self.transaction_cost
        
        if position_size <= 0:
            return
        
        self.capital -= (cost + tx_cost)
        self.position = position_size
        self.entry_price = effective_price
        self.entry_direction = "BUY"
        self.stop_loss_price = effective_price * (1 - self.stop_loss_pct)
        self.take_profit_price = effective_price * (1 + self.take_profit_pct)
    
    def _execute_sell(self, price: float, timestamp, reason: str = "signal"):
        """Close a long position."""
        if self.position <= 0:
            return
        
        effective_price = price * (1 - self.spread / 2)
        revenue = self.position * effective_price
        tx_cost = revenue * self.transaction_cost
        
        pnl = revenue - tx_cost - (self.position * self.entry_price)
        self.capital += revenue - tx_cost
        
        self.trades.append({
            "entry_time": None,  # will be filled by run()
            "exit_time": timestamp,
            "direction": self.entry_direction,
            "entry_price": self.entry_price,
            "exit_price": effective_price,
            "position_size": self.position,
            "pnl": pnl,
            "return_pct": (effective_price / self.entry_price - 1) * 100,
            "reason": reason,
        })
        
        self.risk_manager.update_capital(pnl)
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_direction = None
    
    def _check_stop_loss_take_profit(self, high: float, low: float, timestamp):
        """Check if stop loss or take profit is hit."""
        if self.position <= 0:
            return
        
        if self.entry_direction == "BUY":
            if low <= self.stop_loss_price:
                self._execute_sell(self.stop_loss_price, timestamp, reason="stop_loss")
            elif high >= self.take_profit_price:
                self._execute_sell(self.take_profit_price, timestamp, reason="take_profit")
    
    def run(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """
        Run backtest.
        
        Args:
            df: OHLCV DataFrame.
            signals: Series of 'BUY', 'SELL', 'HOLD' aligned with df index.
        
        Returns:
            Dictionary with performance metrics.
        """
        self.capital = self.initial_capital
        self.position = 0.0
        self.trades = []
        self.equity_curve = []
        self.risk_manager.reset()
        
        entry_time = None
        
        for i in range(len(df)):
            timestamp = df.index[i]
            close = df["Close"].iloc[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else "HOLD"
            
            # Check stop loss / take profit first
            self._check_stop_loss_take_profit(high, low, timestamp)
            
            # Process signal
            if not self.risk_manager.can_trade():
                pass  # Trading halted
            elif signal == "BUY" and self.position <= 0:
                entry_time = timestamp
                self._execute_buy(close, timestamp)
            elif signal == "SELL" and self.position > 0:
                if entry_time and self.trades:
                    pass  # Entry time already captured
                self._execute_sell(close, timestamp, reason="signal")
            
            # Track equity
            position_value = self.position * close if self.position > 0 else 0
            total_equity = self.capital + position_value
            self.equity_curve.append(total_equity)
        
        # Close any remaining position at last price
        if self.position > 0:
            self._execute_sell(df["Close"].iloc[-1], df.index[-1], reason="end_of_data")
        
        # Fill entry times
        trade_idx = 0
        for i, trade in enumerate(self.trades):
            if trade["entry_time"] is None:
                # Find the corresponding entry
                trade["entry_time"] = entry_time if entry_time else trade["exit_time"]
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> dict:
        """Calculate comprehensive performance metrics."""
        if not self.equity_curve:
            return {}
        
        equity = np.array(self.equity_curve)
        
        # Total return
        total_return = (equity[-1] / self.initial_capital - 1) * 100
        
        # Daily returns (from equity curve)
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        
        # Sharpe ratio (annualized, assuming ~252 trading days)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Win rate
        if self.trades:
            wins = sum(1 for t in self.trades if t["pnl"] > 0)
            win_rate = wins / len(self.trades) * 100
        else:
            wins = 0
            win_rate = 0.0
        
        # Profit factor
        gross_profit = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        metrics = {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 3),
            "total_trades": len(self.trades),
            "winning_trades": wins,
            "losing_trades": len(self.trades) - wins,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "final_equity": round(equity[-1], 2),
            "initial_capital": self.initial_capital,
        }
        
        return metrics
    
    def get_equity_curve_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return equity curve as a DataFrame."""
        n = min(len(self.equity_curve), len(df))
        return pd.DataFrame({
            "DateTime": df.index[:n],
            "Equity": self.equity_curve[:n],
        }).set_index("DateTime")
    
    def get_trades_df(self) -> pd.DataFrame:
        """Return trade history as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def print_metrics(self, metrics: dict = None):
        """Pretty-print performance metrics."""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        print(f"\n╔══════════════════════════════════════════╗")
        print(f"║       BACKTEST PERFORMANCE REPORT        ║")
        print(f"╠══════════════════════════════════════════╣")
        print(f"║  Total Return     : {metrics.get('total_return_pct', 0):>10.2f}%         ║")
        print(f"║  Sharpe Ratio     : {metrics.get('sharpe_ratio', 0):>10.3f}          ║")
        print(f"║  Sortino Ratio    : {metrics.get('sortino_ratio', 0):>10.3f}          ║")
        print(f"║  Max Drawdown     : {metrics.get('max_drawdown_pct', 0):>10.2f}%         ║")
        print(f"║  Win Rate         : {metrics.get('win_rate_pct', 0):>10.2f}%         ║")
        print(f"║  Profit Factor    : {metrics.get('profit_factor', 0):>10.3f}          ║")
        print(f"║  Total Trades     : {metrics.get('total_trades', 0):>10d}          ║")
        print(f"║  Final Equity     : ${metrics.get('final_equity', 0):>12,.2f}    ║")
        print(f"╚══════════════════════════════════════════╝")
