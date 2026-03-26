"""
Professional risk management module.
Stop loss, take profit, position sizing, max drawdown protection.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RISK_PER_TRADE, MAX_DRAWDOWN_LIMIT, MAX_PORTFOLIO_EXPOSURE,
    DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT, SPREAD_MAP,
    MDD_HARD_OVERRIDE, KELLY_FRACTION
)


class RiskManager:
    """
    Risk management engine for the trading system.
    """
    
    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = None,
        max_drawdown: float = None,
        max_exposure: float = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.risk_per_trade = risk_per_trade or RISK_PER_TRADE
        self.max_drawdown = max_drawdown or MAX_DRAWDOWN_LIMIT
        self.max_exposure = max_exposure or MAX_PORTFOLIO_EXPOSURE
        self.stop_loss_pct = stop_loss_pct or DEFAULT_STOP_LOSS_PCT
        self.take_profit_pct = take_profit_pct or DEFAULT_TAKE_PROFIT_PCT
        
        self.hard_mdd_limit = MDD_HARD_OVERRIDE
        self.kelly_fraction = KELLY_FRACTION
        
        self.open_positions_value = 0.0
        self.is_halted = False
        self.is_hard_halted = False
    
    def kelly_position_size(self, win_prob: float, win_loss_ratio: float) -> float:
        """Calculate Half-Kelly position fraction."""
        if win_loss_ratio <= 0:
            return 0.0
        f_star = win_prob - ((1 - win_prob) / win_loss_ratio)
        if f_star <= 0:
            return 0.0
        return f_star * self.kelly_fraction

    def calculate_position_size(
        self, entry_price: float, stop_loss_price: float = None,
        use_kelly: bool = False, win_prob: float = 0.53, win_loss_ratio: float = 2.0
    ) -> float:
        """
        Calculate position size based on risk per trade or Kelly criterion.
        
        Risk per trade = risk_pct * current_capital
        Position size = risk_amount / (entry - stop_loss)
        """
        if self.is_halted or self.is_hard_halted:
            return 0.0
        
        if use_kelly:
            kelly_pct = self.kelly_position_size(win_prob, win_loss_ratio)
            risk_amount = kelly_pct * self.current_capital
        else:
            risk_amount = self.risk_per_trade * self.current_capital
        
        if stop_loss_price and entry_price != stop_loss_price:
            risk_per_unit = abs(entry_price - stop_loss_price)
            position_size = risk_amount / risk_per_unit
        else:
            # Default: use stop_loss_pct
            risk_per_unit = entry_price * self.stop_loss_pct
            position_size = risk_amount / risk_per_unit
        
        # Check max exposure
        position_value = position_size * entry_price
        max_position_value = self.current_capital * self.max_exposure - self.open_positions_value
        
        if position_value > max_position_value:
            position_size = max(0, max_position_value / entry_price)
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, direction: str = "BUY") -> float:
        """Calculate stop loss price."""
        if direction == "BUY":
            return entry_price * (1 - self.stop_loss_pct)
        else:  # SELL / SHORT
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, direction: str = "BUY") -> float:
        """Calculate take profit price."""
        if direction == "BUY":
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def update_capital(self, pnl: float):
        """Update capital after a trade closes."""
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self._check_drawdown()
    
    def _check_drawdown(self):
        """Check if max drawdown is breached."""
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if drawdown >= self.max_drawdown:
                self.is_halted = True
                print(f"[RISK] ⚠ Trading HALTED — drawdown {drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
            if drawdown >= self.hard_mdd_limit:
                self.is_hard_halted = True
                print(f"[RISK] ⛔ SYSTEM TERMINATED — hard drawdown limit breached {self.hard_mdd_limit:.2%}")
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown as a fraction."""
        if self.peak_capital > 0:
            return (self.peak_capital - self.current_capital) / self.peak_capital
        return 0.0
    
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self.is_hard_halted:
            return False
        if self.is_halted:
            return False
        if self.open_positions_value >= self.current_capital * self.max_exposure:
            return False
        return True
    
    def get_spread_cost(self, market_type: str, entry_price: float, position_size: float) -> float:
        """Calculate spread cost for a trade."""
        spread_pct = SPREAD_MAP.get(market_type, 0.001)
        return entry_price * position_size * spread_pct
    
    def reset(self):
        """Reset risk manager to initial state."""
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.open_positions_value = 0.0
        self.is_halted = False
