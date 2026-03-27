"""
Append-only JSONL store for agent signals; supports pending resolution rows.

Uses advisory file locking on POSIX for concurrent writers and atomic replace on rewrite.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from agent_consensus.markouts import compute_markouts
from agent_consensus.schema import AgentSignalRecord, AgentVote

try:
    import fcntl  # type: ignore
except ImportError:  # Windows
    fcntl = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SignalLogStore:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def append(self, record: AgentSignalRecord) -> None:
        line = json.dumps(record.to_dict(), default=str) + "\n"
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def append_vote(self, vote: AgentVote) -> AgentSignalRecord:
        rec = AgentSignalRecord.from_vote(vote)
        self.append(rec)
        return rec

    def iter_records(self) -> Iterator[AgentSignalRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                raw = f.read()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield AgentSignalRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Invalid agent signal log line {lineno}: {e}") from e

    def load_all(self) -> List[AgentSignalRecord]:
        return list(self.iter_records())

    def update_row_prices(
        self,
        decision_id: str,
        agent_name: str,
        price_5m: Optional[float] = None,
        price_15m: Optional[float] = None,
    ) -> int:
        """
        Update first matching row; recompute markouts. Atomic replace entire file.

        Returns:
            1 if a row was updated, 0 if no match.
        """
        if price_5m is None and price_15m is None:
            return 0

        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

        if not os.path.exists(self.path):
            return 0

        # Single exclusive lock: read → mutate → truncate+rewrite (no gap for concurrent appends)
        with open(self.path, "r+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                raw = f.read()
                rows: List[AgentSignalRecord] = []
                for lineno, line in enumerate(raw.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(AgentSignalRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                        raise ValueError(f"Invalid agent signal log line {lineno}: {e}") from e

                if not rows:
                    return 0

                updated = 0
                out_lines: List[str] = []
                for r in rows:
                    if (
                        updated == 0
                        and r.decision_id == decision_id
                        and r.agent_name == agent_name
                    ):
                        if price_5m is not None:
                            r.price_5m = float(price_5m)
                        if price_15m is not None:
                            r.price_15m = float(price_15m)
                        compute_markouts(r)
                        if r.price_5m is not None and r.price_15m is not None:
                            r.resolved_at = _utc_now_iso()
                        updated = 1
                    out_lines.append(json.dumps(r.to_dict(), default=str) + "\n")

                body = "".join(out_lines)
                f.seek(0)
                f.write(body)
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
                return updated
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
