from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class MarketRegime(BaseModel):
    regime: str  # e.g., "Trending", "Ranging", "News-Volatile", "Default"
    confidence: float
    reasoning: str

class AgentContext(BaseModel):
    symbol: str
    timeframe: str
    ohlcv_df: Any  # Could be json/dict or pandas DataFrame (using Any for flexibility)
    indicators: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentOutput(BaseModel):
    agent_name: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trigger_price: Optional[float] = None
    valid_until: Optional[datetime] = None

class WeightedConsensus(BaseModel):
    signal: str
    confidence: float
    reasoning: str
    agent_outputs: List[AgentOutput]
    regime: MarketRegime

class TradeExecution(BaseModel):
    symbol: str
    signal: str
    confidence: float
    trigger_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime
