"""Process-local deployment request-rate limiting helpers.

These guards bound request frequency. They are intentionally separate from
subprocess concurrency slots and resource/cost budgets.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from ipaddress import ip_address

from flask import request


_SUBMISSION_PATHS = {
    '/api/v1/eval/invoke': 'eval',
    '/api/v1/eval/launch': 'eval',
    '/api/v1/eval/resume/invoke': 'eval',
    '/api/v1/perf/invoke': 'perf',
    '/api/v1/perf/launch': 'perf',
    '/api/v1/perf/resume/invoke': 'perf',
    '/api/v1/aigc/invoke': 'aigc',
    '/api/v1/eval/batch/launch': 'batch',
    '/api/v1/eval/batch/resume': 'batch',
    '/api/v1/perf/batch/launch': 'batch',
    '/api/v1/perf/batch/resume': 'batch',
}

_RATE_DEFAULTS = {
    'register': (5, 3600),
    'eval': (30, 60),
    'perf': (30, 60),
    'aigc': (10, 60),
    'batch': (10, 60),
}


class SlidingWindowLimiter:
    """Thread-safe in-memory sliding-window limiter.

    State is deliberately process-local: it adds no SQLite write contention to
    hot submission paths. A service restart resets the short request window.
    """

    def __init__(self, *, max_keys: int = 10000):
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def consume(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Consume one event, returning ``(allowed, retry_after_seconds)``."""
        if limit <= 0 or window_seconds <= 0:
            return True, 0
        current = time.monotonic() if now is None else now
        cutoff = current - window_seconds
        with self._lock:
            stamps = self._events.setdefault(key, deque())
            while stamps and stamps[0] <= cutoff:
                stamps.popleft()
            if len(stamps) >= limit:
                return False, max(1, math.ceil(stamps[0] + window_seconds - current))
            stamps.append(current)
            while len(self._events) > self._max_keys:
                oldest_key = min(
                    self._events,
                    key=lambda event_key: self._events[event_key][-1],
                )
                self._events.pop(oldest_key, None)
            return True, 0


def get_client_ip() -> str:
    """Resolve the nearest untrusted client IP from a trusted proxy chain."""
    trusted_proxies = {
        value.strip()
        for value in os.environ.get('TRUSTED_PROXIES', '127.0.0.1,::1').split(',')
        if value.strip()
    }
    remote = request.remote_addr or ''
    if remote not in trusted_proxies:
        return remote or '-'

    forwarded = [value.strip() for value in request.headers.get('X-Forwarded-For', '').split(',')]
    chain = [value for value in forwarded if value]
    real_ip = request.headers.get('X-Real-IP', '').strip()
    if not chain and real_ip:
        chain = [real_ip]

    # Work from the server side of the chain. Trusted proxies are peeled off;
    # the nearest valid non-proxy address is the client. This is safe whether
    # the proxy overwrites XFF or appends to an incoming header.
    for candidate in reversed(chain):
        try:
            normalized = str(ip_address(candidate))
        except ValueError:
            continue
        if normalized not in trusted_proxies:
            return normalized
    return remote or '-'


def classify_submission_path(path: str, method: str) -> str | None:
    """Return the task-submission rate bucket for an exact route path."""
    if method.upper() != 'POST':
        return None
    return _SUBMISSION_PATHS.get(path)


def get_rate_policy(bucket: str) -> tuple[int, int]:
    """Read a validated request count/window pair from deployment settings."""
    if bucket not in _RATE_DEFAULTS:
        raise ValueError(f'Unknown rate-limit bucket: {bucket}')
    default_requests, default_window = _RATE_DEFAULTS[bucket]
    prefix = f'RATE_LIMIT_{bucket.upper()}'
    try:
        requests = int(os.environ.get(f'{prefix}_REQUESTS', str(default_requests)))
        window = int(os.environ.get(f'{prefix}_WINDOW_SECONDS', str(default_window)))
    except ValueError as exc:
        raise ValueError(f'{prefix}_REQUESTS and {prefix}_WINDOW_SECONDS must be integers') from exc
    if requests < 0 or window < 0:
        raise ValueError(f'{prefix}_REQUESTS and {prefix}_WINDOW_SECONDS must be non-negative')
    return requests, window
