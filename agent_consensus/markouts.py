"""
Markout math: signed log-return aligned with agent direction.
"""
from __future__ import annotations

import math
from typing import Optional

from agent_consensus.schema import AgentSignalRecord, Direction


def directional_score(direction: Direction, entry: float, future_price: Optional[float]) -> Optional[float]:
    """
    If future_price is None, return None.
    If direction == 0 (neutral), return None — no directional markout to score.
    Else return direction * log(future / entry).
    """
    if future_price is None:
        return None
    if entry is None or entry <= 0 or future_price <= 0:
        return None
    if direction == 0:
        return None
    return float(direction) * math.log(future_price / entry)


def compute_markouts(rec: AgentSignalRecord) -> AgentSignalRecord:
    """Fill markout_5m / markout_15m from price fields (mutates and returns same record)."""
    rec.markout_5m = directional_score(rec.direction, rec.entry_price, rec.price_5m)
    rec.markout_15m = directional_score(rec.direction, rec.entry_price, rec.price_15m)
    return rec
