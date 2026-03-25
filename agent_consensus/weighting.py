"""
Capped, renormalized weights from performance state.

Keeps math out of the LLM: feed these numbers into the Manager prompt or multiply votes.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from agent_consensus.performance_tracker import AgentPerformanceState


def combined_score(
    st: AgentPerformanceState,
    win_weight: float = 1.0,
    markout_weight: float = 10.0,
) -> float:
    """
    Markouts are small (log-return scale); scale up so they compete with 0..1 win rate.
    """
    return win_weight * st.ema_win_15m + markout_weight * st.ema_markout_15m


def _validate_caps(n_agents: int, min_weight: float, max_weight: float) -> None:
    if n_agents <= 0:
        return
    if not (0 < min_weight <= 1 and 0 < max_weight <= 1):
        raise ValueError("min_weight and max_weight must be in (0, 1]")
    if min_weight > max_weight:
        raise ValueError("min_weight cannot exceed max_weight")
    # Feasibility: need room on simplex after enforcing minimum mass per agent
    if n_agents * min_weight > 1.0 + 1e-12:
        raise ValueError(
            f"Infeasible caps: n_agents={n_agents} * min_weight={min_weight} > 1. "
            "Lower min_weight or reduce committee size."
        )
    # Need at least one feasible assignment: max_weight * n >= 1 is sufficient when min*n<=1
    if max_weight * n_agents < 1.0 - 1e-12:
        raise ValueError(
            f"Infeasible caps: max_weight={max_weight} too small for n_agents={n_agents} "
            "(cannot sum to 1)."
        )


def _stable_softmax_weighted(scores: List[Tuple[str, float]], temperature: float) -> Dict[str, float]:
    """exp(t * s_i) normalized; subtract max exponent for numerical stability."""
    if not scores:
        return {}
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    names = [n for n, _ in scores]
    raw = [s for _, s in scores]
    m = max(raw)
    exps = [math.exp(temperature * (s - m)) for s in raw]
    z = sum(exps) or 1.0
    return {names[i]: exps[i] / z for i in range(len(names))}


def compute_weights(
    states: Iterable[AgentPerformanceState],
    *,
    temperature: float = 4.0,
    min_weight: float = 0.15,
    max_weight: float = 0.55,
    win_weight: float = 1.0,
    markout_weight: float = 10.0,
    agent_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Stable softmax over combined score, then clip to [min_weight, max_weight], then renormalize.

    If `agent_names` is provided, any missing agent gets a neutral synthetic state so
    new agents are not dropped from the committee.

    Raises ValueError if cap constraints cannot be satisfied on the probability simplex.
    """
    state_map: Dict[str, AgentPerformanceState] = {s.agent_name: s for s in states}
    names = list(agent_names) if agent_names is not None else sorted(state_map.keys())
    if not names:
        return {}

    if len(names) == 1:
        return {names[0]: 1.0}

    _validate_caps(len(names), min_weight, max_weight)

    scored: List[Tuple[str, float]] = []
    for name in names:
        st = state_map.get(name)
        if st is None:
            st = AgentPerformanceState(agent_name=name)
        sc = combined_score(st, win_weight=win_weight, markout_weight=markout_weight)
        scored.append((name, sc))

    weights = _stable_softmax_weighted(scored, temperature)

    clipped = {n: min(max(weights[n], min_weight), max_weight) for n in names}
    s = sum(clipped.values())
    if s <= 0:
        u = 1.0 / len(clipped)
        return {n: u for n in clipped}
    return {n: w / s for n, w in clipped.items()}
