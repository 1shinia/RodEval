import os
from collections import deque

from evalscope.constants import DEFAULT_WORK_DIR

OUTPUT_DIR = os.path.abspath(os.getenv('EVALSCOPE_OUTPUT_DIR', DEFAULT_WORK_DIR))


def validate_task_id(task_id: str) -> None:
    """Validate a task_id value.

    Raises:
        ValueError: if task_id is empty, too long, or contains path-traversal characters.
    """
    if not task_id:
        raise ValueError('task_id is required')
    if len(task_id) > 255:
        raise ValueError('task_id is too long')
    if '\x00' in task_id:
        raise ValueError('Invalid task_id')
    if os.path.basename(task_id) != task_id:
        raise ValueError('Invalid task_id')


def validate_root_path(root: str) -> str:
    """Validate that *root* resolves to a path within OUTPUT_DIR.

    Returns the resolved absolute path.
    Raises ValueError if the path escapes the allowed directory.
    """
    resolved = os.path.realpath(root)
    allowed = os.path.realpath(OUTPUT_DIR)
    # The resolved path must be exactly OUTPUT_DIR or a subdirectory of it
    if resolved != allowed and not resolved.startswith(allowed + os.sep):
        raise ValueError(f'root_path must be within {OUTPUT_DIR}')
    return resolved


def validate_report_name(report_name: str, root: str) -> str:
    """Validate that *report_name* resolves to a path within *root*.

    Uses ``process_report_name`` to extract the directory prefix, then
    checks that the resolved absolute path does not escape *root*.

    Returns the resolved absolute path of the report directory.
    Raises ValueError if the report_name format is invalid or the path
    escapes the allowed directory.
    """
    from evalscope.utils.data_utils import process_report_name

    try:
        prefix, _, _ = process_report_name(report_name)
    except (ValueError, IndexError) as exc:
        raise ValueError('Invalid report_name format') from exc

    resolved = os.path.realpath(os.path.join(root, prefix))
    root_resolved = os.path.realpath(root)
    if resolved != root_resolved and not resolved.startswith(root_resolved + os.sep):
        raise ValueError('Invalid report_name: path escapes output directory')
    return resolved


def create_log_file(task_id: str, sub_path: str) -> str:
    """Create an empty log file for a given task so that log polling does not raise FileNotFoundError.

    Returns the absolute path of the created log file.
    """
    validate_task_id(task_id)

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8'):
            pass
    return log_file


def get_log_content(task_id: str, sub_path: str, start_line: int = None, page: int = 500) -> dict:
    """Read log content for a given task with pagination support.

    Args:
        task_id: The task identifier.
        sub_path: The log file path relative to task output directory.
        start_line: If None, read last `page` lines from end; otherwise read from this line (must be >= 0).
        page: Number of lines to read (must be >= 1, default 500).

    Returns:
        dict with keys:
            - text: log content (lines kept as-is, preserving original newlines)
            - head_line: 0-based start line number of returned content
            - tail_line: 0-based end line number (exclusive)
            - total_lines: total line count of the log file
    """
    validate_task_id(task_id)

    # Validate parameters
    if page < 1:
        raise ValueError('page must be >= 1')
    if start_line is not None and start_line < 0:
        raise ValueError('start_line must be >= 0')

    log_file = os.path.join(OUTPUT_DIR, task_id, sub_path)
    if not os.path.exists(log_file):
        return {'text': '', 'head_line': 0, 'tail_line': 0, 'total_lines': 0}

    with open(log_file, 'r', encoding='utf-8') as f:
        # Single-pass: count lines and collect requested lines
        # This ensures total_lines matches Python's line iteration semantics
        total_lines = 0
        lines = deque(maxlen=page) if start_line is None else []

        for line in f:
            total_lines += 1
            if start_line is None:
                # deque with maxlen automatically keeps last N lines (O(1))
                lines.append(line)
            else:
                if total_lines > start_line and len(lines) < page:
                    lines.append(line)

        # Compute head_line based on actual total_lines
        if start_line is None:
            head_line = max(0, total_lines - page)
            lines = list(lines)
        elif start_line >= total_lines:
            return {'text': '', 'head_line': total_lines, 'tail_line': total_lines, 'total_lines': total_lines}
        else:
            head_line = start_line

    tail_line = head_line + len(lines)
    # Use ''.join to preserve original newlines in each line
    return {'text': ''.join(lines), 'head_line': head_line, 'tail_line': tail_line, 'total_lines': total_lines}


# Retention policy for old task logs (default 30 days, configurable via env var).
_RETENTION_DAYS = int(os.getenv('EVALSCOPE_LOG_RETENTION_DAYS', '30'))


def cleanup_old_task_logs() -> dict:
    """Remove task output directories older than the retention period.

    Uses SQLite metadata (``finished_at``) when available; falls back to
    directory modification time.  Skips directories that are currently
    running (checked against the in-memory registry).

    Returns:
        dict with keys ``removed`` (int), ``skipped_running`` (int),
        ``freed_bytes`` (int), ``errors`` (list of str).
    """
    import shutil
    import time

    from evalscope.utils.logger import get_logger

    logger = get_logger()
    cutoff = time.time() - _RETENTION_DAYS * 86400
    removed = 0
    skipped_running = 0
    freed_bytes = 0
    errors: list = []

    if not os.path.isdir(OUTPUT_DIR):
        return {'removed': 0, 'skipped_running': 0, 'freed_bytes': 0, 'errors': ['output_dir not found']}

    # Gather running task IDs from the in-memory registry.
    running_ids: set = set()
    try:
        from ..utils.process import get_running_tasks as _get_running
        for t in _get_running():
            running_ids.add(t['task_id'])
    except Exception:
        pass

    # Try SQLite for finished_at timestamps.
    db_finished: dict = {}
    try:
        from .. import db as _db
        conn = _db._get_conn()
        rows = conn.execute('SELECT task_id, finished_at FROM task_state WHERE finished_at IS NOT NULL').fetchall()
        db_finished = {r['task_id']: r['finished_at'] for r in rows}
    except Exception:
        pass

    for entry in os.scandir(OUTPUT_DIR):
        if not entry.is_dir():
            continue
        dir_name = entry.name

        # Never clean up the service log or metadata DB.
        if dir_name.startswith('evalscope_'):
            continue

        # Skip running tasks.
        if dir_name in running_ids:
            skipped_running += 1
            continue

        # Determine age: prefer SQLite finished_at, fall back to mtime.
        if dir_name in db_finished:
            try:
                # finished_at is ISO format: '2026-06-01T12:00:00'
                ts = time.mktime(time.strptime(db_finished[dir_name][:19], '%Y-%m-%dT%H:%M:%S'))
            except Exception:
                ts = entry.stat().st_mtime
        else:
            ts = entry.stat().st_mtime

        if ts >= cutoff:
            continue  # still within retention window

        # Remove the directory.
        try:
            dir_size = _dir_size(entry.path)
            shutil.rmtree(entry.path)
            removed += 1
            freed_bytes += dir_size
        except Exception as e:
            errors.append(f'{dir_name}: {e}')

    if removed > 0:
        logger.info(
            'Log retention cleanup: removed %d old task directories, freed %.1f MB (%d running tasks skipped)', removed,
            freed_bytes / (1024 * 1024), skipped_running
        )
    if errors:
        for err in errors:
            logger.warning('Log retention cleanup error: %s', err)

    return {
        'removed': removed,
        'skipped_running': skipped_running,
        'freed_bytes': freed_bytes,
        'errors': errors,
    }


def _dir_size(path: str) -> int:
    """Return total size of a directory tree in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total
