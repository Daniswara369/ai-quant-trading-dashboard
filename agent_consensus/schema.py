"""
JSON-serializable records for agent telemetry and markout resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import uuid
from datetime import datetime, timezone


Direction = Literal[-1, 0, 1]

REQUIRED_RECORD_KEYS = frozenset(
    {
        "decision_id",
        "timestamp",
        "agent_name",
        "direction",
        "confidence",
        "entry_price",
        "symbol",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_direction(value: Any) -> Direction:
    """Coerce JSON/int/str to -1, 0, or 1."""
    if isinstance(value, str):
        value = value.strip()
        if value in ("+", "1", "bull", "long", "buy"):
            return 1
        if value in ("-", "-1", "bear", "short", "sell"):
            return -1
        if value in ("0", "neutral", "hold", "flat", ""):
            return 0
    v = int(value)
    if v not in (-1, 0, 1):
        raise ValueError(f"direction must be -1, 0, or 1; got {v!r}")
    return v  # type: ignore[return-value]


def clamp_confidence(x: float) -> float:
    c = float(x)
    if c != c:  # NaN
        raise ValueError("confidence must be a finite number")
    if c < 0.0 or c > 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return c


def _opt_float(x) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    if v != v:  # NaN
        return None
    return v


@dataclass
class AgentVote:
    """Single agent output at decision time."""

    agent_name: str
    direction: Direction
    confidence: float  # 0..1
    entry_price: float  # mid or last
    symbol: str
    timeframe: str = "1h"
    weight: Optional[float] = None  # Added for reporting
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not self.agent_name or not str(self.agent_name).strip():
            raise ValueError("agent_name is required")
        self.agent_name = str(self.agent_name).strip()
        self.direction = normalize_direction(self.direction)
        self.confidence = clamp_confidence(self.confidence)
        ep = float(self.entry_price)
        if ep != ep or ep <= 0:
            raise ValueError("entry_price must be a finite positive number")
        self.entry_price = ep
        if not self.symbol or not str(self.symbol).strip():
            raise ValueError("symbol is required")
        self.symbol = str(self.symbol).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSignalRecord:
    """
    One logged row per agent vote, extended when 5m/15m prices are known.

    Markouts are *signed* log-returns in the direction of the agent's call:
      markout = direction * log(price_horizon / entry_price)
    For direction 0 (neutral), markouts stay None (no directional edge to score).
    """

    decision_id: str
    timestamp: str
    agent_name: str
    direction: Direction
    confidence: float
    entry_price: float
    symbol: str
    timeframe: str
    weight: Optional[float] = None  # Added for reporting
    price_5m: Optional[float] = None
    price_15m: Optional[float] = None
    markout_5m: Optional[float] = None
    markout_15m: Optional[float] = None
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_vote(cls, vote: AgentVote) -> AgentSignalRecord:
        return cls(
            decision_id=vote.decision_id,
            timestamp=vote.timestamp,
            agent_name=vote.agent_name,
            direction=vote.direction,
            confidence=vote.confidence,
            entry_price=vote.entry_price,
            symbol=vote.symbol,
            timeframe=vote.timeframe,
            weight=vote.weight,
        )

    @classmethod
    def from_dict(cls, d: dict) -> AgentSignalRecord:
        missing = REQUIRED_RECORD_KEYS - d.keys()
        if missing:
            raise KeyError(f"AgentSignalRecord missing keys: {sorted(missing)}")
        return cls(
            decision_id=str(d["decision_id"]),
            timestamp=str(d["timestamp"]),
            agent_name=str(d["agent_name"]).strip(),
            direction=normalize_direction(d["direction"]),
            confidence=clamp_confidence(d["confidence"]),
            entry_price=float(d["entry_price"]),
            symbol=str(d["symbol"]).strip(),
            timeframe=str(d.get("timeframe") or "1h"),
            weight=_opt_float(d.get("weight")),
            price_5m=_opt_float(d.get("price_5m")),
            price_15m=_opt_float(d.get("price_15m")),
            markout_5m=_opt_float(d.get("markout_5m")),
            markout_15m=_opt_float(d.get("markout_15m")),
            resolved_at=d.get("resolved_at"),
        )
