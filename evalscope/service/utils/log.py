import os
from collections import deque

from evalscope.constants import DEFAULT_WORK_DIR

# Default to the project root's outputs/ directory, not CWD-sensitive ./outputs.
# Falls back to DEFAULT_WORK_DIR if the package directory can't be resolved.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_default_output = os.path.join(_project_root, 'outputs') if os.path.isdir(_project_root) else DEFAULT_WORK_DIR
OUTPUT_DIR = os.path.abspath(os.getenv('EVALSCOPE_OUTPUT_DIR', _default_output))


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


def validate_root_path(root: str, allowed_root: str | None = None) -> str:
    """Validate that *root* resolves inside the configured output root.

    ``allowed_root`` lets service instances started with a custom ``--outputs``
    directory validate against that runtime root instead of the module-level
    default.  Callers that omit it retain the historical ``OUTPUT_DIR`` policy.
    """
    resolved = os.path.realpath(root)
    allowed = os.path.realpath(allowed_root or OUTPUT_DIR)
    if resolved != allowed and not resolved.startswith(allowed + os.sep):
        raise ValueError(f'root_path must be within {allowed}')
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
    if not os.path.isdir(resolved):
        raise ValueError(f'Report not found: {prefix}')
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


# Retention policy for incomplete task logs (default 7 days, configurable via env var).
# Completed evaluations (progress.json status == 'completed') are kept forever.
_RETENTION_DAYS = int(os.getenv('EVALSCOPE_LOG_RETENTION_DAYS', '7'))


def _read_progress_status(dir_path: str) -> tuple[str, str]:
    """Read progress.json and return ``(status, updated_at)``.

    Returns ``('', '')`` if progress.json is missing or malformed.
    """
    import json
    pj = os.path.join(dir_path, 'progress.json')
    if not os.path.isfile(pj):
        return '', ''
    try:
        with open(pj, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('status', ''), data.get('updated_at', '')
    except (json.JSONDecodeError, IOError, OSError, ValueError):
        return '', ''


def _parse_timestamp(value: str) -> float | None:
    """Parse an ISO timestamp into epoch seconds, or None if unparseable."""
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def cleanup_old_task_logs() -> dict:
    """Remove incomplete task output directories older than the retention period.

    Completed evaluations (``progress.json`` status == ``completed``) are kept
    indefinitely.  Incomplete tasks (stopped / failed / error / running /
    missing progress.json) are removed after ``_RETENTION_DAYS``.

    Age is determined by ``progress.json`` ``updated_at`` when available,
    falling back to directory modification time.  Directories that are
    currently running (in-memory registry) are skipped.

    Returns:
        dict with keys ``removed`` (int), ``skipped_running`` (int),
        ``skipped_completed`` (int), ``freed_bytes`` (int), ``errors`` (list).
    """
    import shutil
    import time

    from evalscope.utils.logger import get_logger

    logger = get_logger()
    cutoff = time.time() - _RETENTION_DAYS * 86400
    removed = 0
    skipped_running = 0
    skipped_completed = 0
    freed_bytes = 0
    errors: list = []

    if not os.path.isdir(OUTPUT_DIR):
        return {
            'removed': 0,
            'skipped_running': 0,
            'skipped_completed': 0,
            'freed_bytes': 0,
            'errors': ['output_dir not found'],
        }

    # Gather running task IDs from the in-memory registry.
    running_ids: set = set()
    try:
        from ..utils.process import get_running_tasks as _get_running
        for t in _get_running():
            running_ids.add(t['task_id'])
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

        # Completed evaluations are kept forever.
        status, updated_at = _read_progress_status(entry.path)
        if status == 'completed':
            skipped_completed += 1
            continue

        # Perf tasks never write progress.json; a finished run is marked by
        # its generated report. Treat it as completed so the retention
        # cleanup keeps perf results instead of deleting them after 7 days.
        if os.path.isfile(os.path.join(entry.path, 'perf', 'perf_report.html')):
            skipped_completed += 1
            continue

        # Determine age: prefer progress.json updated_at, fall back to mtime.
        ts = _parse_timestamp(updated_at)
        if ts is None:
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
            'Log retention cleanup: removed %d incomplete task directories, '
            'freed %.1f MB (kept %d completed, %d running skipped)',
            removed,
            freed_bytes / (1024 * 1024),
            skipped_completed,
            skipped_running,
        )
    if errors:
        for err in errors:
            logger.warning('Log retention cleanup error: %s', err)

    return {
        'removed': removed,
        'skipped_running': skipped_running,
        'skipped_completed': skipped_completed,
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
