"""
Rolling EMA stats per agent from resolved markouts.

Uses 15m markout as primary edge signal; optional 5m for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from dateutil import parser as date_parser

from agent_consensus.schema import AgentSignalRecord


@dataclass
class AgentPerformanceState:
    """Smoothed per-agent telemetry."""

    agent_name: str
    ema_win_15m: float = 0.5  # prior: neutral coin flip
    ema_markout_15m: float = 0.0
    n_resolved: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "ema_win_15m": self.ema_win_15m,
            "ema_markout_15m": self.ema_markout_15m,
            "n_resolved": self.n_resolved,
        }


def _sort_key_ts(ts: str) -> float:
    try:
        return date_parser.parse(ts).timestamp()
    except (ValueError, TypeError, OverflowError):
        return 0.0


class PerformanceTracker:
    """
    Online update: feed resolved records in chronological order for correct EMA.

    Win definition (directional bets only): markout_15m > 0 means the signed log-return
    aligned with the agent's direction was positive.

    Neutral (direction 0) rows are skipped — they do not advance n_resolved.
    """

    def __init__(self, alpha: float = 0.15):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._agents: Dict[str, AgentPerformanceState] = {}

    def _state(self, name: str) -> AgentPerformanceState:
        if name not in self._agents:
            self._agents[name] = AgentPerformanceState(agent_name=name)
        return self._agents[name]

    def update_from_record(self, rec: AgentSignalRecord) -> None:
        if rec.direction == 0:
            return
        if rec.markout_15m is None:
            return
        st = self._state(rec.agent_name)
        st.n_resolved += 1
        win = 1.0 if rec.markout_15m > 0 else 0.0
        a = self.alpha
        st.ema_win_15m = a * win + (1 - a) * st.ema_win_15m
        st.ema_markout_15m = a * rec.markout_15m + (1 - a) * st.ema_markout_15m

    def ingest(self, records: Iterable[AgentSignalRecord]) -> None:
        resolved = [r for r in records if r.direction != 0 and r.markout_15m is not None]
        resolved.sort(key=lambda r: _sort_key_ts(r.timestamp))
        for r in resolved:
            self.update_from_record(r)

    def snapshot(self) -> Dict[str, AgentPerformanceState]:
        return dict(self._agents)

    def all_states(self) -> List[AgentPerformanceState]:
        return list(self._agents.values())

    def reset(self) -> None:
        self._agents.clear()
