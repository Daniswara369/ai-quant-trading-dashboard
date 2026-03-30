"""
Committee-level agreement metrics (meta-layer input for Manager / Gemini).

Pure math: complements performance-weighted voting without replacing it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from agent_consensus.schema import Direction

Vote = Tuple[Direction, float]  # direction, confidence


def committee_agreement_metrics(votes: List[Vote]) -> Dict[str, float]:
    """
    Summarize how aligned the committee is on direction, confidence-weighted.

    Returns:
        n_votes: count of directional (non-zero) votes
        mass_up / mass_down: sum of confidences per side
        balance: |up - down| / (up + down) in [0, 1] — 1 = unanimous, 0 = split mass
        net_direction: sign of (mass_up - mass_down), -1 / 0 / 1
    """
    directional = [(d, c) for d, c in votes if d != 0]
    n = len(directional)
    if n == 0:
        return {
            "n_votes": 0,
            "mass_up": 0.0,
            "mass_down": 0.0,
            "balance": 0.0,
            "net_direction": 0,
        }

    mass_up = sum(c for d, c in directional if d > 0)
    mass_down = sum(c for d, c in directional if d < 0)
    denom = mass_up + mass_down
    balance = abs(mass_up - mass_down) / denom if denom > 0 else 0.0
    net = mass_up - mass_down
    if net > 1e-12:
        net_dir = 1
    elif net < -1e-12:
        net_dir = -1
    else:
        net_dir = 0

    return {
        "n_votes": float(n),
        "mass_up": round(mass_up, 6),
        "mass_down": round(mass_down, 6),
        "balance": round(balance, 6),
        "net_direction": net_dir,
    }
