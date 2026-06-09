"""
In-memory run store for SSE streaming debug API.

Stores prepared runs with TTL and provides lookup by run_id.

Design:
  - Process-level dict: run_id → RunEntry
  - Each run stores: run_id, agent_id, request, status, created_at
  - Simple TTL: entries older than 10 minutes are auto-pruned on access
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from app.schemas.consultation import ConsultationRequest

TTL_SECONDS = 600  # 10 minutes — more than enough for a single request


@dataclass
class RunEntry:
    run_id: str
    agent_id: str
    request: ConsultationRequest
    created_at: float = field(default_factory=time.time)
    status: str = "prepared"  # prepared / streaming / done / error

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def expired(self) -> bool:
        return self.age_seconds > TTL_SECONDS


# ── In-process storage ───────────────────────────────────────────

_PENDING_RUNS: dict[str, RunEntry] = {}


def _prune_expired() -> None:
    """Remove expired runs."""
    expired_ids = [rid for rid, entry in _PENDING_RUNS.items() if entry.expired]
    for rid in expired_ids:
        del _PENDING_RUNS[rid]


def create_run(agent_id: str, request: ConsultationRequest) -> RunEntry:
    """Create a new run and store it."""
    _prune_expired()
    run_id = uuid.uuid4().hex
    entry = RunEntry(run_id=run_id, agent_id=agent_id, request=request)
    _PENDING_RUNS[run_id] = entry
    return entry


def get_run(run_id: str) -> RunEntry | None:
    """Look up a run by ID. Returns None if expired or not found."""
    _prune_expired()
    entry = _PENDING_RUNS.get(run_id)
    if entry is None:
        return None
    if entry.expired:
        _PENDING_RUNS.pop(run_id, None)
        return None
    return entry


def mark_run(run_id: str, status: str) -> None:
    """Update run status."""
    entry = _PENDING_RUNS.get(run_id)
    if entry:
        entry.status = status


def delete_run(run_id: str) -> None:
    """Remove a run after streaming completes."""
    _PENDING_RUNS.pop(run_id, None)
