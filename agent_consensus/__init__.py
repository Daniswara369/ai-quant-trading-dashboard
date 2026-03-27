"""
Agent signal logging, markout resolution, rolling performance, and capped weights.

Designed to sit alongside an LLM "Manager" (e.g. Gemini): keep scoring deterministic in Python.
"""

from agent_consensus.schema import AgentSignalRecord, AgentVote, normalize_direction
from agent_consensus.signal_log import SignalLogStore
from agent_consensus.markouts import compute_markouts, directional_score
from agent_consensus.performance_tracker import PerformanceTracker
from agent_consensus.weighting import combined_score, compute_weights
from agent_consensus.agreement import committee_agreement_metrics
from agent_consensus.snapshot import build_committee_snapshot

__all__ = [
    "AgentSignalRecord",
    "AgentVote",
    "normalize_direction",
    "SignalLogStore",
    "compute_markouts",
    "directional_score",
    "PerformanceTracker",
    "combined_score",
    "compute_weights",
    "committee_agreement_metrics",
    "build_committee_snapshot",
]
