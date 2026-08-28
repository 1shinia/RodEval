"""Task-artifact authorization policy independent of Flask transport details.

Keeping metadata/owner-marker lookup here prevents report, SSE and generated-file
routes from each reimplementing a subtly different ownership rule.  The module
is deliberately small so the metadata backend can later be swapped from SQLite
without changing every blueprint.
"""

import os
from typing import Iterable


def task_artifact_owned_by(
    task_id: str,
    tables: Iterable[str],
    *,
    user_id: int,
    is_admin: bool,
    output_dir: str,
) -> bool:
    """Return whether ``user_id`` may read artifacts belonging to ``task_id``.

    Metadata rows are authoritative.  If a task has not yet been indexed (for
    example while it is starting), the durable ``.owner`` marker is used.
    Legacy directories with neither metadata nor marker are admin-only.
    """
    from .db import _TASK_ID_TABLES, _get_conn
    from .utils.log import validate_task_id

    try:
        validate_task_id(task_id)
    except ValueError:
        return False

    conn = _get_conn()
    for table in tables:
        if table not in _TASK_ID_TABLES:
            raise ValueError(f'Unsupported task table: {table}')
        row = conn.execute(f'SELECT user_id FROM {table} WHERE task_id = ?', (task_id,)).fetchone()
        if row is not None:
            return int(row[0]) == int(user_id)

    marker = os.path.join(str(output_dir), task_id, '.owner')
    try:
        with open(marker, encoding='utf-8') as f:
            return int(f.read().strip()) == int(user_id)
    except (OSError, ValueError):
        return bool(is_admin)
