"""
High-level helpers: load log → EMA performance → capped weights (+ optional agreement).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from agent_consensus.agreement import Vote, committee_agreement_metrics
from agent_consensus.performance_tracker import AgentPerformanceState, PerformanceTracker
from agent_consensus.signal_log import SignalLogStore
from agent_consensus.weighting import compute_weights


def build_committee_snapshot(
    store: SignalLogStore,
    committee_agent_names: List[str],
    *,
    alpha: float = 0.15,
    temperature: float = 4.0,
    min_weight: float = 0.15,
    max_weight: float = 0.55,
    latest_votes: Optional[List[Vote]] = None,
) -> Dict:
    """
    Produce weights + per-agent EMA stats for Manager prompt injection.

    latest_votes: optional (direction, confidence) tuples for the *current* bar only,
    used only to compute agreement metrics (not persisted here).
    """
    rows = store.load_all()
    tracker = PerformanceTracker(alpha=alpha)
    tracker.ingest(rows)
    states = tracker.all_states()
    # Ensure every committee member appears in output even with zero history
    state_by_name = {s.agent_name: s for s in states}
    perf_out = []
    for name in committee_agent_names:
        st = state_by_name.get(name) or AgentPerformanceState(agent_name=name)
        perf_out.append(st.to_dict())

    weights = compute_weights(
        states,
        agent_names=committee_agent_names,
        temperature=temperature,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    out: Dict = {
        "weights": weights,
        "performance": perf_out,
        "n_log_rows": len(rows),
    }
    if latest_votes is not None:
        out["agreement"] = committee_agreement_metrics(latest_votes)
    return out
