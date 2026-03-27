"""FastAPI routes for agent committee (uses tempfile log path)."""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

pytest.importorskip("api.server")
import config as app_config  # noqa: E402
from api.server import app  # noqa: E402


@pytest.fixture()
def client_and_log(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)
    monkeypatch.setattr(app_config, "AGENT_SIGNAL_LOG_PATH", path)
    yield TestClient(app), path
    if os.path.exists(path):
        os.unlink(path)


def test_snapshot_vote_resolve_agreement(client_and_log):
    client, path = client_and_log
    r = client.get("/api/agent-committee/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["n_log_rows"] == 0
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-9

    r2 = client.post(
        "/api/agent-committee/vote",
        json={
            "agent_name": "TechnicalAnalyst",
            "direction": 1,
            "confidence": 0.8,
            "entry_price": 100.0,
            "symbol": "BTCUSDT",
        },
    )
    assert r2.status_code == 200
    did = r2.json()["record"]["decision_id"]

    r3 = client.post(
        "/api/agent-committee/resolve-markout",
        json={
            "decision_id": did,
            "agent_name": "TechnicalAnalyst",
            "price_5m": 101.0,
            "price_15m": 102.0,
        },
    )
    assert r3.status_code == 200

    r4 = client.get("/api/agent-committee/snapshot")
    assert r4.status_code == 200
    assert r4.json()["n_log_rows"] == 1

    r5 = client.post(
        "/api/agent-committee/agreement",
        json={
            "votes": [
                {"direction": 1, "confidence": 0.7},
                {"direction": -1, "confidence": 0.7},
            ]
        },
    )
    assert r5.status_code == 200
    assert r5.json()["balance"] == pytest.approx(0.0)


def test_resolve_markout_404(client_and_log):
    client, _ = client_and_log
    r = client.post(
        "/api/agent-committee/resolve-markout",
        json={
            "decision_id": "nonexistent-id",
            "agent_name": "TechnicalAnalyst",
            "price_5m": 100.0,
            "price_15m": 101.0,
        },
    )
    assert r.status_code == 404


def test_vote_invalid_direction_422(client_and_log):
    client, _ = client_and_log
    r = client.post(
        "/api/agent-committee/vote",
        json={
            "agent_name": "TechnicalAnalyst",
            "direction": 2,
            "confidence": 0.5,
            "entry_price": 1.0,
            "symbol": "BTCUSDT",
        },
    )
    assert r.status_code == 422


def test_snapshot_corrupt_log_500(client_and_log, monkeypatch):
    client, path = client_and_log
    with open(path, "w") as f:
        f.write("{not valid json\n")
    r = client.get("/api/agent-committee/snapshot")
    assert r.status_code == 500
    assert "line 1" in r.json()["detail"]


def test_vote_empty_symbol_422(client_and_log):
    client, _ = client_and_log
    r = client.post(
        "/api/agent-committee/vote",
        json={
            "agent_name": "TechnicalAnalyst",
            "direction": 1,
            "confidence": 0.5,
            "entry_price": 1.0,
            "symbol": "   ",
        },
    )
    assert r.status_code == 422
