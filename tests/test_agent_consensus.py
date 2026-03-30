"""
Full tests for agent telemetry: schema, markouts, log store, EMA, weights, agreement, snapshot.
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

from agent_consensus import (
    AgentVote,
    SignalLogStore,
    PerformanceTracker,
    compute_markouts,
    compute_weights,
    directional_score,
    committee_agreement_metrics,
    build_committee_snapshot,
    normalize_direction,
)
from agent_consensus.performance_tracker import AgentPerformanceState
from agent_consensus.schema import AgentSignalRecord


def test_normalize_direction_strings():
    assert normalize_direction("1") == 1
    assert normalize_direction("buy") == 1
    assert normalize_direction("-1") == -1
    assert normalize_direction("neutral") == 0


def test_agent_vote_validation():
    v = AgentVote(
        agent_name="TechnicalAnalyst",
        direction=1,
        confidence=0.8,
        entry_price=100.0,
        symbol="BTCUSDT",
    )
    assert v.direction == 1
    with pytest.raises(ValueError):
        AgentVote("x", 2, 0.5, 1.0, "BTCUSDT")
    with pytest.raises(ValueError):
        AgentVote("x", 1, 1.5, 1.0, "BTCUSDT")
    with pytest.raises(ValueError):
        AgentVote("x", 1, 0.5, -1.0, "BTCUSDT")


def test_directional_score_neutral_none():
    assert directional_score(0, 100.0, 101.0) is None
    assert directional_score(1, 100.0, None) is None


def test_directional_score_long():
    s = directional_score(1, 100.0, 110.0)
    assert s is not None and s > 0


def test_markout_compute():
    r = AgentSignalRecord(
        decision_id="d1",
        timestamp="2026-01-01T00:00:00+00:00",
        agent_name="A",
        direction=1,
        confidence=0.7,
        entry_price=100.0,
        symbol="BTCUSDT",
        timeframe="1h",
        price_5m=101.0,
        price_15m=100.0,
    )
    compute_markouts(r)
    assert r.markout_5m == pytest.approx(math.log(1.01))
    assert r.markout_15m == pytest.approx(math.log(1.0))


def test_signal_log_roundtrip_and_resolve():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        store = SignalLogStore(path)
        did = "dec-1"
        v = AgentVote(
            agent_name="RiskAuditor",
            direction=-1,
            confidence=0.6,
            entry_price=200.0,
            symbol="ETHUSDT",
            decision_id=did,
        )
        store.append_vote(v)
        assert store.update_row_prices(did, "RiskAuditor", price_5m=198.0) == 1
        rows = store.load_all()
        assert len(rows) == 1
        assert rows[0].price_5m == 198.0
        assert rows[0].price_15m is None
        assert rows[0].resolved_at is None
        assert store.update_row_prices(did, "RiskAuditor", price_15m=199.0) == 1
        rows = store.load_all()
        assert rows[0].price_15m == 199.0
        assert rows[0].resolved_at is not None
        assert rows[0].markout_15m is not None
    finally:
        os.unlink(path)


def test_performance_tracker_skips_neutral():
    pt = PerformanceTracker(alpha=0.5)
    neutral = AgentSignalRecord(
        decision_id="n1",
        timestamp="2026-01-01T00:00:00+00:00",
        agent_name="X",
        direction=0,
        confidence=0.5,
        entry_price=1.0,
        symbol="S",
        timeframe="1h",
        price_5m=1.1,
        price_15m=1.2,
    )
    compute_markouts(neutral)
    assert neutral.markout_15m is None
    pt.ingest([neutral])
    assert pt.all_states() == []


def test_performance_tracker_ema_ordering():
    pt = PerformanceTracker(alpha=0.5)
    r1 = AgentSignalRecord(
        "1",
        "2026-01-01T00:00:00+00:00",
        "A",
        1,
        1.0,
        100.0,
        "BTCUSDT",
        "1h",
        price_5m=101.0,
        price_15m=102.0,
    )
    compute_markouts(r1)
    r2 = AgentSignalRecord(
        "2",
        "2026-01-02T00:00:00+00:00",
        "A",
        1,
        1.0,
        100.0,
        "BTCUSDT",
        "1h",
        price_5m=99.0,
        price_15m=98.0,
    )
    compute_markouts(r2)
    pt.ingest([r2, r1])
    st = pt.snapshot()["A"]
    assert st.n_resolved == 2


def test_compute_weights_single_agent():
    st = AgentPerformanceState("Only", ema_win_15m=0.9, ema_markout_15m=0.01, n_resolved=5)
    w = compute_weights([st], agent_names=["Only"])
    assert w == {"Only": 1.0}


def test_compute_weights_infeasible_min_raises():
    names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    states = [AgentPerformanceState(n) for n in names]
    with pytest.raises(ValueError, match="Infeasible"):
        compute_weights(states, agent_names=names, min_weight=0.2)


def test_compute_weights_stable_softmax_no_inf():
    states = [
        AgentPerformanceState("A", ema_win_15m=0.99, ema_markout_15m=0.05, n_resolved=100),
        AgentPerformanceState("B", ema_win_15m=0.1, ema_markout_15m=-0.05, n_resolved=100),
    ]
    w = compute_weights(
        states,
        agent_names=["A", "B"],
        temperature=100.0,
        min_weight=0.15,
        max_weight=0.85,
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(math.isfinite(v) for v in w.values())


def test_committee_agreement():
    m = committee_agreement_metrics([(1, 0.8), (-1, 0.8)])
    assert m["balance"] == pytest.approx(0.0)
    m2 = committee_agreement_metrics([(1, 0.9), (1, 0.1)])
    assert m2["balance"] == pytest.approx(1.0)


def test_build_committee_snapshot():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        store = SignalLogStore(path)
        did = "d-collective"
        for name, direction in [
            ("TechnicalAnalyst", 1),
            ("SentimentStrategist", 1),
            ("RiskAuditor", -1),
        ]:
            v = AgentVote(
                agent_name=name,
                direction=direction,
                confidence=0.7,
                entry_price=50.0,
                symbol="BTCUSDT",
                decision_id=did,
            )
            store.append_vote(v)
            store.update_row_prices(did, name, price_5m=51.0, price_15m=52.0)
        committee = ["TechnicalAnalyst", "SentimentStrategist", "RiskAuditor"]
        snap = build_committee_snapshot(
            store,
            committee,
            latest_votes=[(1, 0.7), (1, 0.6), (-1, 0.5)],
        )
        assert snap["n_log_rows"] == 3
        assert "weights" in snap and "agreement" in snap
        assert abs(sum(snap["weights"].values()) - 1.0) < 1e-6
    finally:
        os.unlink(path)


def test_corrupt_jsonl_raises():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write("not json\n")
        store = SignalLogStore(path)
        with pytest.raises(ValueError, match="line 1"):
            store.load_all()
    finally:
        os.unlink(path)
