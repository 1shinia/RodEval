"""Atomic progress terminal-state persistence shared by task backends."""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from .time_utils import utc_now_iso


_progress_lock = threading.Lock()


def write_terminal_progress(
    path: str,
    status: str,
    *,
    pipeline: str,
    error: str = '',
    extra: dict[str, Any] | None = None,
) -> None:
    """Merge and atomically persist a canonical terminal progress state.

    A user stop is authoritative: a late worker completion/failure in the same
    service process must not overwrite it.
    """
    with _progress_lock:
        data: dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if data.get('status') in {'stopped', 'cancelled'} and status != 'stopped':
            return
        data.update({
            'status': status,
            'phase': status,
            'pipeline': pipeline,
            'updated_at': utc_now_iso(),
        })
        if status == 'completed':
            data['percent'] = 100.0
        if error:
            data['error'] = error
        if extra:
            data.update(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
