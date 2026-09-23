"""Canonical task and batch lifecycle states.

This module is the single backend authority for lifecycle semantics.  Legacy
wire values are accepted at boundaries and normalized before persistence or
presentation.
"""

from __future__ import annotations

from typing import Final

QUEUED: Final = 'queued'
STARTING: Final = 'starting'
RUNNING: Final = 'running'
CANCELLING: Final = 'cancelling'
COMPLETED: Final = 'completed'
PARTIAL_SUCCESS: Final = 'partial_success'
FAILED: Final = 'failed'
STOPPED: Final = 'stopped'
ORPHANED: Final = 'orphaned'

ACTIVE_STATES: Final = frozenset({QUEUED, STARTING, RUNNING, CANCELLING})
TERMINAL_STATES: Final = frozenset({
    COMPLETED, PARTIAL_SUCCESS, FAILED, STOPPED, ORPHANED,
})
ALL_STATES: Final = ACTIVE_STATES | TERMINAL_STATES

# Kept only at compatibility boundaries.  New writes must use canonical names.
LEGACY_STATE_ALIASES: Final = {
    'cancelled': STOPPED,
    'error': FAILED,
    'ok': COMPLETED,
    'success': COMPLETED,
    'completed_with_warnings': PARTIAL_SUCCESS,
}


def normalize_status(status: str | None, *, default: str = FAILED) -> str:
    value = str(status or '').strip().lower()
    value = LEGACY_STATE_ALIASES.get(value, value)
    if value not in ALL_STATES:
        return default
    return value


def normalize_persisted_task_status(status: str | None) -> str:
    """Map canonical lifecycle values onto the v20 ``task_state`` CHECK.

    The runtime table predates queued/starting/partial-success.  Keep its
    append-only schema compatible while exposing the richer status on APIs.
    """
    value = normalize_status(status)
    if value in {QUEUED, STARTING, CANCELLING}:
        return RUNNING
    if value == PARTIAL_SUCCESS:
        return COMPLETED
    return value


def is_active(status: str | None) -> bool:
    return normalize_status(status) in ACTIVE_STATES


def is_terminal(status: str | None) -> bool:
    return normalize_status(status) in TERMINAL_STATES


def aggregate_batch_status(*, total: int, completed: int, errors: int, stopped: bool = False) -> str:
    """Derive a batch terminal state from row outcomes.

    ``completed`` counts successful rows only; ``errors`` counts real failures.
    A user stop wins over the aggregate while work remains resumable.
    """
    if stopped:
        return STOPPED
    if errors and completed:
        return PARTIAL_SUCCESS
    if errors:
        return FAILED
    if total > 0 and completed >= total:
        return COMPLETED
    return RUNNING
