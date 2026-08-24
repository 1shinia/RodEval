"""SQLite metadata store for eval reports and perf tasks.

Provides fast listing/filtering without scanning the filesystem on every
request.  The database file lives at ``{OUTPUT_DIR}/evalscope_meta.db``.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any

from evalscope.utils.logger import get_logger

logger = get_logger()

_local = threading.local()
_db_path: str | None = None

# Process-wide write lock. All SQLite write operations (upserts, deletes,
# task-state mutations) go through this lock so that concurrent request
# threads (waitress workers) cannot deadlock on the database write lock
# when multiple tasks complete at the same time. Reads are lock-free.
_write_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Schema versioning — simple linear migration system
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 8  # Bump when adding migrations below

# Each migration: (target_version, description, SQL statements)
# Migrations are applied in order; only those with version > current are run.
_MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1, 'initial schema + indexes', '''
        CREATE TABLE IF NOT EXISTS eval_reports (
            task_id        TEXT PRIMARY KEY,
            model_name     TEXT NOT NULL,
            dataset_name   TEXT NOT NULL,
            score          REAL DEFAULT 0,
            num_samples    INTEGER DEFAULT 0,
            timestamp      TEXT,
            dataset_scores TEXT
        );
        CREATE TABLE IF NOT EXISTS perf_tasks (
            task_id    TEXT PRIMARY KEY,
            model      TEXT NOT NULL,
            api        TEXT DEFAULT '',
            dataset    TEXT DEFAULT '',
            runs       INTEGER DEFAULT 0,
            has_report INTEGER DEFAULT 0,
            timestamp  TEXT
        );
        CREATE TABLE IF NOT EXISTS task_state (
            task_id    TEXT PRIMARY KEY,
            task_type  TEXT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'running',
            pid        INTEGER,
            model      TEXT DEFAULT '',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_eval_reports_model
            ON eval_reports(model_name);
        CREATE INDEX IF NOT EXISTS idx_eval_reports_dataset
            ON eval_reports(dataset_name);
        CREATE INDEX IF NOT EXISTS idx_eval_reports_timestamp
            ON eval_reports(timestamp);
        CREATE INDEX IF NOT EXISTS idx_eval_reports_score
            ON eval_reports(score);
        CREATE INDEX IF NOT EXISTS idx_perf_tasks_model
            ON perf_tasks(model);
        CREATE INDEX IF NOT EXISTS idx_perf_tasks_dataset
            ON perf_tasks(dataset);
        CREATE INDEX IF NOT EXISTS idx_perf_tasks_timestamp
            ON perf_tasks(timestamp);
        CREATE INDEX IF NOT EXISTS idx_task_state_status
            ON task_state(status);
        CREATE INDEX IF NOT EXISTS idx_task_state_task_type
            ON task_state(task_type);
    '''
    ),
    (
        2, 'add perf_tasks extra columns', '''
        -- Example future migration: add columns to perf_tasks
        -- ALTER TABLE perf_tasks ADD COLUMN concurrency INTEGER DEFAULT 1;
        -- ALTER TABLE perf_tasks ADD COLUMN duration_seconds REAL DEFAULT 0;
        -- (No-op for now — placeholder showing the pattern)
        SELECT 1;
    '''
    ),
    (
        3, 'add eval_backend column', '''
        ALTER TABLE eval_reports ADD COLUMN eval_backend TEXT DEFAULT '';
        UPDATE eval_reports SET eval_backend = '' WHERE eval_backend IS NULL;
    '''
    ),
    (
        4, 'add compare_reports table', '''
        CREATE TABLE IF NOT EXISTS compare_reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            task_ids    TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            task_count  INTEGER DEFAULT 0
        );
    '''
    ),
    (
        5, 'add backend and root_path to compare_reports', '''
        ALTER TABLE compare_reports ADD COLUMN backend TEXT DEFAULT 'Perf';
        ALTER TABLE compare_reports ADD COLUMN root_path TEXT DEFAULT '';
        UPDATE compare_reports SET backend = 'Perf' WHERE backend IS NULL;
        UPDATE compare_reports SET root_path = '' WHERE root_path IS NULL;
    '''
    ),
    (
        6, 'add users table + user_id columns', '''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role          TEXT DEFAULT 'user',
            created_at    TEXT NOT NULL
        );
    '''
    ),
    (
        7, 'add user_id to data tables', '''
        ALTER TABLE eval_reports ADD COLUMN user_id INTEGER DEFAULT 1;
        ALTER TABLE perf_tasks ADD COLUMN user_id INTEGER DEFAULT 1;
        ALTER TABLE compare_reports ADD COLUMN user_id INTEGER DEFAULT 1;
        UPDATE eval_reports SET user_id = 1 WHERE user_id IS NULL;
        UPDATE perf_tasks SET user_id = 1 WHERE user_id IS NULL;
        UPDATE compare_reports SET user_id = 1 WHERE user_id IS NULL;
    '''
    ),
    (
        8, 'add token_blacklist table', '''
        CREATE TABLE IF NOT EXISTS token_blacklist (
            jti         TEXT PRIMARY KEY,
            expires_at  TEXT NOT NULL
        );
    '''
    ),
]


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version (0 if no version table exists)."""
    try:
        row = conn.execute('SELECT version FROM schema_version ORDER BY version DESC LIMIT 1').fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply pending migrations to bring the schema up to SCHEMA_VERSION."""
    # Ensure version tracking table exists
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at  TEXT NOT NULL
        )
    '''
    )
    conn.commit()

    current = _get_schema_version(conn)
    if current >= SCHEMA_VERSION:
        return

    for version, description, sql in _MIGRATIONS:
        if version <= current:
            continue
        if version > SCHEMA_VERSION:
            break
        logger.info(f'DB migration v{current}→v{version}: {description}')
        conn.executescript(sql)
        conn.execute(
            'INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)',
            (version, description, datetime.now().isoformat()),
        )
        conn.commit()
        current = version

    logger.info(f'DB schema at v{current}')


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def init_db(output_dir: str) -> None:
    """Initialise the database path and create tables if needed."""
    global _db_path
    _db_path = os.path.join(output_dir, 'evalscope_meta.db')
    os.makedirs(output_dir, exist_ok=True)
    conn = _get_conn()
    _migrate(conn)
    logger.info(f'SQLite metadata DB ready: {_db_path}')


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection."""
    if _db_path is None:
        raise RuntimeError('init_db() has not been called')
    conn: sqlite3.Connection | None = getattr(_local, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(_db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        # Aggressive auto-checkpoint: flush WAL after ~800 KB instead of 4 MB default
        conn.execute('PRAGMA wal_autocheckpoint=200')
        _local.conn = conn
    return conn


def checkpoint_db() -> dict:
    """Force a WAL checkpoint to truncate the write-ahead log.

    Returns a dict with ``busy``, ``log``, ``checkpointed`` page counts.
    Call after bulk writes (e.g. backfill) or periodically on a busy server.
    Never raises: a failed checkpoint (e.g. a reader holding a snapshot)
    is a non-fatal condition — the WAL is simply kept for later.
    """
    try:
        conn = _get_conn()
        # TRUNCATE: reset the WAL file to zero bytes after checkpoint
        row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        result = {'busy': row[0], 'log': row[1], 'checkpointed': row[2]}
        if result['checkpointed'] > 0:
            logger.debug(
                'WAL checkpoint: %d pages moved, %d in log, %d busy', result['checkpointed'], result['log'], result['busy']
            )
        return result
    except Exception as e:
        logger.warning(f'WAL checkpoint skipped (non-fatal): {e}')
        return {'busy': -1, 'log': 0, 'checkpointed': 0}


# ---------------------------------------------------------------------------
# Eval reports CRUD
# ---------------------------------------------------------------------------


def _compute_total_num(report_list) -> int:
    """Sample count for a report list, shared by live eval and backfill.

    Multi-metric reports (e.g. vendor verifiers) carry one entry per metric
    whose ``num`` is the count of rows that triggered that validator (0 = not
    triggered).  Summing them over-counts and treating any 0 row as "full
    dataset" would wrongly yield the -1 sentinel.  Return the max positive
    count, or -1 (全量, limits was null) only when nothing is > 0.
    """
    positive = [r.num for r in report_list if (r.num or 0) > 0]
    return max(positive) if positive else -1


def upsert_eval_report(
    task_id: str,
    model_name: str,
    dataset_name: str,
    score: float,
    num_samples: int,
    timestamp: str,
    dataset_scores: dict | None = None,
    eval_backend: str = '',
    user_id: int = 1,
) -> None:
    with _write_lock:
        conn = _get_conn()
        for attempt in range(5):
            try:
                conn.execute(
                    '''INSERT OR REPLACE INTO eval_reports
                       (task_id, model_name, dataset_name, score, num_samples, timestamp, dataset_scores, eval_backend, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        task_id, model_name, dataset_name, score, num_samples, timestamp,
                        json.dumps(dataset_scores, ensure_ascii=False) if dataset_scores else None,
                        eval_backend,
                        user_id,
                    ),
                )
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                # Roll back any half-open implicit transaction so the
                # thread-local connection isn't left holding a stale write
                # intent after a busy/lock failure.
                try:
                    conn.rollback()
                except Exception:
                    pass
                if 'locked' in str(e) and attempt < 4:
                    import time
                    time.sleep(1 * (attempt + 1))
                    continue
                raise


def query_eval_reports(
    search: str = '',
    models: str = '',
    datasets: str = '',
    score_min: float | None = None,
    score_max: float | None = None,
    sort_by: str = 'time',
    sort_order: str = 'desc',
    page: int = 1,
    page_size: int = 20,
    backend: str = '',
    user_id: int = 1,
) -> tuple[list[dict], int, list[str], list[str]]:
    """Return ``(items, total, available_models, available_datasets)``."""
    conn = _get_conn()
    where: list[str] = ['user_id = ?']
    params: list[Any] = [user_id]

    if search:
        # SQLite LIKE is case-insensitive for ASCII — no LOWER() needed, index-friendly
        where.append('(model_name LIKE ? OR dataset_name LIKE ?)')
        params.extend([f'%{search}%', f'%{search}%'])
    if models:
        model_set = [m.strip().lower() for m in models.split(';') if m.strip()]
        if model_set:
            # Individual comparisons with COLLATE NOCASE (IN doesn't support it)
            model_conds = []
            for m in model_set:
                model_conds.append('model_name = ? COLLATE NOCASE')
                params.append(m)
            where.append(f'({" OR ".join(model_conds)})')
    if datasets:
        ds_set = [d.strip().lower() for d in datasets.split(';') if d.strip()]
        if ds_set:
            ds_conds = []
            for ds in ds_set:
                ds_conds.append('dataset_name LIKE ?')
                params.append(f'%{ds}%')
            where.append(f'({" OR ".join(ds_conds)})')
    if score_min is not None:
        where.append('score >= ?')
        params.append(score_min)
    if score_max is not None:
        where.append('score <= ?')
        params.append(score_max)
    if backend:
        where.append('eval_backend = ?')
        params.append(backend)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ''

    sort_map = {
        'score': 'score',
        'model': 'model_name',
        'dataset': 'dataset_name',
        'time': 'timestamp',
    }
    col = sort_map.get(sort_by, 'timestamp')
    direction = 'DESC' if sort_order == 'desc' else 'ASC'

    # Available filter values (before filtering) — scoped to backend
    _backend_where = 'AND eval_backend = ?' if backend else ''
    _backend_params = [backend] if backend else []
    avail_models = [
        r[0] for r in conn.
        execute(f'SELECT DISTINCT model_name FROM eval_reports WHERE model_name != "" {_backend_where} ORDER BY model_name',
                _backend_params).fetchall()
    ]
    avail_datasets_raw = conn.execute(
        f'SELECT DISTINCT dataset_name FROM eval_reports WHERE dataset_name != "" {_backend_where}',
        _backend_params,
    ).fetchall()
    avail_datasets: list[str] = []
    for r in avail_datasets_raw:
        for d in r[0].split(', '):
            d = d.strip()
            if d and d not in avail_datasets:
                avail_datasets.append(d)
    avail_datasets.sort()

    total = conn.execute(f'SELECT COUNT(*) FROM eval_reports {where_sql}', params).fetchone()[0]

    offset = (max(1, page) - 1) * page_size
    rows = conn.execute(
        f'''SELECT task_id, model_name, dataset_name, score, num_samples,
                   timestamp, dataset_scores
            FROM eval_reports {where_sql}
            ORDER BY {col} {direction}
            LIMIT ? OFFSET ?''',
        [*params, page_size, offset],
    ).fetchall()

    items: list[dict] = []
    for row in rows:
        ds_scores = None
        if row['dataset_scores']:
            try:
                ds_scores = json.loads(row['dataset_scores'])
            except (json.JSONDecodeError, TypeError):
                pass
        # Construct the full report_name format expected by process_report_name:
        # {task_id}@@{model_name}::{dataset_name}
        # But only if task_id doesn't already contain the format
        task_id = row['task_id']
        if '@@' in task_id:
            # Already in full format, use as-is
            report_name = task_id
        else:
            # Construct full format
            report_name = f"{task_id}@@{row['model_name']}::{row['dataset_name']}"
        items.append({
            'name': report_name,
            'model_name': row['model_name'],
            'dataset_name': row['dataset_name'],
            'score': row['score'],
            'num_samples': row['num_samples'],
            'timestamp': row['timestamp'],
            'dataset_scores': ds_scores,
        })

    return items, total, avail_models, avail_datasets


def delete_eval_report(task_id: str, user_id: int | None = None) -> None:
    with _write_lock:
        conn = _get_conn()
        if user_id is not None:
            conn.execute('DELETE FROM eval_reports WHERE task_id = ? AND user_id = ?', (task_id, user_id))
        else:
            conn.execute('DELETE FROM eval_reports WHERE task_id = ?', (task_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Perf tasks CRUD
# ---------------------------------------------------------------------------


def upsert_perf_task(
    task_id: str,
    model: str,
    api: str,
    dataset: str,
    runs: int,
    has_report: bool,
    timestamp: str,
    user_id: int = 1,
) -> None:
    with _write_lock:
        conn = _get_conn()
        for attempt in range(5):
            try:
                conn.execute(
                    '''INSERT OR REPLACE INTO perf_tasks
                       (task_id, model, api, dataset, runs, has_report, timestamp, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task_id, model, api, dataset, runs, int(has_report), timestamp, user_id),
                )
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if 'locked' in str(e) and attempt < 4:
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise


def cleanup_perf_tasks(output_dir: str, user_id: int | None = None) -> int:
    """Remove perf_tasks rows whose directories no longer exist on disk.

    Returns the number of rows removed.
    """
    with _write_lock:
        conn = _get_conn()
        if user_id is not None:
            rows = conn.execute('SELECT task_id FROM perf_tasks WHERE user_id = ?', (user_id,)).fetchall()
        else:
            rows = conn.execute('SELECT task_id FROM perf_tasks').fetchall()
        stale: list[str] = []
        for (tid,) in rows:
            if not os.path.isdir(os.path.join(output_dir, tid)):
                stale.append(tid)
        if stale:
            conn.executemany('DELETE FROM perf_tasks WHERE task_id = ?', [(t,) for t in stale])
            conn.commit()
        return len(stale)


def cleanup_eval_reports(output_dir: str, user_id: int | None = None) -> int:
    """Remove eval_reports rows whose directories no longer exist on disk.

    Returns the number of rows removed.
    """
    with _write_lock:
        conn = _get_conn()
        if user_id is not None:
            rows = conn.execute('SELECT task_id FROM eval_reports WHERE user_id = ?', (user_id,)).fetchall()
        else:
            rows = conn.execute('SELECT task_id FROM eval_reports').fetchall()
        stale: list[str] = []
        for (tid,) in rows:
            if not os.path.isdir(os.path.join(output_dir, tid)):
                stale.append(tid)
        if stale:
            conn.executemany('DELETE FROM eval_reports WHERE task_id = ?', [(t,) for t in stale])
            conn.commit()
        return len(stale)


def upsert_aigc_audio_report(output_dir: str, task_id: str, user_id: int = 1) -> bool:
    """Read AIGC/Audio results.json and upsert into eval_reports.

    Returns True if a report was upserted, False if skipped.
    """
    results_file = os.path.join(output_dir, task_id, 'results.json')
    if not os.path.isfile(results_file):
        return False
    try:
        with open(results_file, encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    model_type = data.get('model_type', '')
    tool = data.get('tool', '')
    model_name = data.get('model', 'unknown')
    # Defensive: cap model_name length to prevent log-corruption spills
    MAX_MODEL_NAME = 200
    if isinstance(model_name, str) and len(model_name) > MAX_MODEL_NAME:
        logger.warning(
            f'Model name too long ({len(model_name)} chars) for {task_id}, '
            f'truncating to {MAX_MODEL_NAME} chars'
        )
        model_name = model_name[:MAX_MODEL_NAME]
    num_samples = data.get('num_samples', 0)
    metrics = data.get('metrics', {})
    if not isinstance(metrics, dict):
        metrics = {}

    # Determine backend
    if model_type in ('txt2img', 'txt2video', 'img2img'):
        eval_backend = 'AIGC_EVAL'
    elif tool in ('asr', 'tts'):
        eval_backend = 'AudioEval'
    else:
        return False  # Not AIGC/Audio

    # Determine primary score
    score = 0.0
    if eval_backend == 'AIGC_EVAL':
        score = metrics.get('clip_score_mean') or 0.0
    elif tool == 'asr':
        wer = metrics.get('wer')
        score = (1.0 - wer) if wer is not None else 0.0
    elif tool == 'tts':
        wer = metrics.get('wer_avg')
        score = (1.0 - wer) if wer is not None else 0.0

    # Timestamp
    ts = data.get('timestamp')
    if ts:
        timestamp = datetime.fromtimestamp(float(ts)).isoformat()
    else:
        try:
            timestamp = datetime.fromtimestamp(os.path.getmtime(results_file)).isoformat()
        except OSError:
            timestamp = ''

    dataset_name = model_type or tool or 'unknown'

    upsert_eval_report(
        task_id=task_id,
        model_name=model_name,
        dataset_name=dataset_name,
        score=round(score, 4),
        num_samples=num_samples,
        timestamp=timestamp,
        dataset_scores=None,
        eval_backend=eval_backend,
        user_id=user_id,
    )
    return True


def query_perf_tasks(
    search: str = '',
    filter_model: str = '',
    filter_dataset: str = '',
    sort_by: str = 'time',
    sort_order: str = 'desc',
    page: int = 1,
    page_size: int = 20,
    user_id: int = 1,
) -> tuple[list[dict], int, list[str], list[str]]:
    """Return ``(items, total, available_models, available_datasets)``."""
    conn = _get_conn()
    where: list[str] = ['user_id = ?']
    params: list[Any] = [user_id]

    if search:
        where.append('(model LIKE ? OR dataset LIKE ?)')
        params.extend([f'%{search}%', f'%{search}%'])
    if filter_model:
        where.append('model = ?')
        params.append(filter_model)
    if filter_dataset:
        where.append('dataset = ?')
        params.append(filter_dataset)

    where_sql = f'WHERE {" AND ".join(where)}' if where else ''

    if sort_by == 'model':
        order_col = 'model'
    else:
        order_col = 'timestamp'
    direction = 'DESC' if sort_order == 'desc' else 'ASC'

    # Available filter values
    avail_models = [
        r[0] for r in conn.
        execute('SELECT DISTINCT model FROM perf_tasks WHERE model != "" AND model != "N/A" ORDER BY model').fetchall()
    ]
    avail_datasets = [
        r[0] for r in conn.execute(
            'SELECT DISTINCT dataset FROM perf_tasks WHERE dataset != "" AND dataset != "N/A" ORDER BY dataset'
        ).fetchall()
    ]

    total = conn.execute(f'SELECT COUNT(*) FROM perf_tasks {where_sql}', params).fetchone()[0]

    offset = (max(1, page) - 1) * page_size
    rows = conn.execute(
        f'''SELECT task_id, model, api, dataset, runs, has_report, timestamp
            FROM perf_tasks {where_sql}
            ORDER BY {order_col} {direction}
            LIMIT ? OFFSET ?''',
        [*params, page_size, offset],
    ).fetchall()

    items: list[dict] = []
    for row in rows:
        items.append({
            'task_id': row['task_id'],
            'model': row['model'],
            'api': row['api'],
            'dataset': row['dataset'],
            'runs': row['runs'],
            'has_report': bool(row['has_report']),
            'timestamp': row['timestamp'],
        })

    return items, total, avail_models, avail_datasets


def delete_perf_task(task_id: str, user_id: int | None = None) -> None:
    with _write_lock:
        conn = _get_conn()
        if user_id is not None:
            conn.execute('DELETE FROM perf_tasks WHERE task_id = ? AND user_id = ?', (task_id, user_id))
        else:
            conn.execute('DELETE FROM perf_tasks WHERE task_id = ?', (task_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Task state (running/completed/failed) — for persistence across restarts
# ---------------------------------------------------------------------------


def upsert_task_state(
    task_id: str,
    task_type: str,
    status: str,
    pid: int | None = None,
    model: str = '',
) -> None:
    """Insert or update a task's runtime state.

    Status values: 'running', 'completed', 'failed', 'stopped', 'orphaned'.
    """
    with _write_lock:
        conn = _get_conn()
        now = datetime.now().isoformat()
        conn.execute(
            '''INSERT INTO task_state (task_id, task_type, status, pid, model, started_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                   status = excluded.status,
                   pid = excluded.pid,
                   updated_at = excluded.updated_at''',
            (task_id, task_type, status, pid, model, now, now),
        )
        conn.commit()


def delete_task_state(task_id: str) -> None:
    with _write_lock:
        conn = _get_conn()
        conn.execute('DELETE FROM task_state WHERE task_id = ?', (task_id, ))
        conn.commit()


def list_running_tasks() -> list[dict]:
    """Return all tasks with status='running'.

    As a safety net, any task whose child PID is no longer alive is
    automatically marked 'orphaned' and excluded from the result.
    This catches edge cases where the subprocess died without the
    parent updating task_state (e.g. OOM kill, segfault).
    """
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, started_at, updated_at
           FROM task_state WHERE status = 'running'
           ORDER BY started_at DESC'''
    ).fetchall()
    alive = []
    now = datetime.now().isoformat()
    for r in rows:
        pid = r['pid']
        if pid and _pid_alive(pid):
            alive.append(dict(r))
        else:
            conn.execute(
                "UPDATE task_state SET status = 'orphaned', updated_at = ? WHERE task_id = ?",
                (now, r['task_id']),
            )
            logger.info(f"Auto-orphaned zombie task {r['task_id']} (PID {pid} dead)")
    if alive and len(alive) < len(rows):
        conn.commit()
    return alive


def get_all_task_states() -> list[dict]:
    """Return all task states (for debugging / admin)."""
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, started_at, updated_at
           FROM task_state ORDER BY started_at DESC'''
    ).fetchall()
    return [dict(r) for r in rows]


def recover_stale_tasks() -> list[str]:
    """Mark 'running' tasks from a previous service instance as 'orphaned'.

    Called on server startup to clean up stale state from a previous crash.
    Uses a PID file (``evalscope_service.pid`` in the outputs directory) to
    determine whether the previous service instance is still alive.  If the
    old service is dead, all running tasks are marked orphaned regardless
    of their child-process liveness (eval children use os.setsid() and can
    outlive the parent service).

    Returns the list of task_ids that were marked orphaned.
    """
    conn = _get_conn()

    if _db_path is None:
        return []

    pid_file = os.path.join(os.path.dirname(_db_path), 'evalscope_service.pid')
    old_pid = _read_service_pid(pid_file)

    # If old service is still alive, its tasks are legitimate — skip recovery.
    if old_pid is not None and _pid_alive(old_pid):
        logger.info(f'Previous service (PID {old_pid}) is still running — skipping stale task recovery.')
        return []

    rows = conn.execute("SELECT task_id FROM task_state WHERE status = 'running'").fetchall()
    if not rows:
        return []

    orphaned = [row['task_id'] for row in rows]
    now = datetime.now().isoformat()
    for tid in orphaned:
        conn.execute(
            "UPDATE task_state SET status = 'orphaned', updated_at = ? WHERE task_id = ?",
            (now, tid),
        )
    conn.commit()
    logger.info(f'Recovered {len(orphaned)} stale tasks from dead service (PID {old_pid}): {orphaned}')
    return orphaned


def write_service_pid(output_dir: str) -> None:
    """Write the current process PID to ``evalscope_service.pid``.

    Must be called once on service startup, before :func:`recover_stale_tasks`.
    """
    pid_file = os.path.join(output_dir, 'evalscope_service.pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))


def _read_service_pid(pid_file: str) -> int | None:
    """Read a PID from *pid_file*; return None if the file is missing or corrupt."""
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with *pid* exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    # Exclude zombies (PID exists but process is terminated, unreaped)
    try:
        with open(f'/proc/{pid}/status', 'r') as f:
            first_line = f.readline()
            if 'zombie' in first_line.lower() or first_line.startswith('State:\tZ'):
                return False
    except (FileNotFoundError, PermissionError):
        pass
    return True


# ---------------------------------------------------------------------------
# Backfill — populate DB from existing filesystem data on first startup
# ---------------------------------------------------------------------------


def _parse_mteb_results_dir(work_dir: str) -> list:
    """Parse MTEB JSON results from a RAG task's results/ directory.

    Mirrors ``evalscope.service.blueprints.eval._parse_mteb_results`` so the
    backfill path indexes RAG tasks identically to live evaluation.  Returns
    a list of SimpleNamespace objects with model_name/dataset_name/score/num.
    """
    from types import SimpleNamespace

    results_dir = os.path.join(work_dir, 'results')
    reports: list = []
    if not os.path.isdir(results_dir):
        return reports

    # Sample count: prefer eval_config.eval.limits, else -1 (full dataset)
    num_samples = -1
    try:
        import yaml as _yaml
        config_path = os.path.join(work_dir, 'configs', 'task_config.yaml')
        if os.path.isfile(config_path):
            with open(config_path) as cf:
                cfg = _yaml.safe_load(cf) or {}
            limits = cfg.get('eval_config', {}).get('eval', {}).get('limits')
            if limits is not None:
                num_samples = int(limits)
    except Exception:
        pass

    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith('.json') or fname == 'model_meta.json':
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                task_name = data.get('task_name', fname.replace('.json', ''))
                scores = data.get('scores', {})
                # Model name from path: results/eval__model_name/master/...
                rel_path = os.path.relpath(fpath, results_dir)
                parts = rel_path.split(os.sep)
                model_name = parts[0].replace('eval__', '') if parts else 'unknown'
                # main_score from first split (usually 'test')
                main_score = None
                for split_data in scores.values():
                    if isinstance(split_data, list) and split_data:
                        main_score = split_data[0].get('main_score')
                        break
                if main_score is not None:
                    reports.append(
                        SimpleNamespace(
                            model_name=model_name,
                            dataset_name=task_name,
                            score=main_score,
                            num=num_samples,
                        )
                    )
            except Exception as e:
                logger.warning(f'Backfill: failed to parse MTEB result {fpath}: {e}')
    return reports


def _parse_clip_results_dir(work_dir: str) -> list:
    """Parse CLIP Benchmark JSON results from <task_dir>/<model_name>/*.json.

    Mirrors ``evalscope.service.blueprints.eval._parse_clip_results`` so the
    backfill path indexes CLIP tasks identically to live evaluation.  Returns
    a list of SimpleNamespace objects with model_name/dataset_name/score/num.
    """
    from types import SimpleNamespace

    reports: list = []
    if not os.path.isdir(work_dir):
        return reports

    _SKIP_DIRS = {'configs', 'logs', 'predictions', 'reviews', 'reports', 'results'}
    for model_dir_name in sorted(os.listdir(work_dir)):
        model_dir = os.path.join(work_dir, model_dir_name)
        if not os.path.isdir(model_dir) or model_dir_name.startswith('.') or model_dir_name in _SKIP_DIRS:
            continue
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(model_dir, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metrics = data.get('metrics') or {}
                score = None
                for key in ('acc1', 'mean_average_precision', 'recall_at_1', 'recall_at_5', 'acc5'):
                    if key in metrics and isinstance(metrics[key], (int, float)) and not isinstance(metrics[key], bool):
                        score = float(metrics[key])
                        break
                if score is None:
                    for v in metrics.values():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            score = float(v)
                            break
                if score is None:
                    logger.warning(f'Backfill: skip CLIP result {fpath}: no numeric metric in {metrics}')
                    continue
                if score > 1:
                    score = score / 100
                reports.append(
                    SimpleNamespace(
                        model_name=data.get('model', model_dir_name),
                        dataset_name=data.get('dataset', fname.replace('.json', '')),
                        score=score,
                        num=data.get('num_samples', -1),
                    )
                )
            except Exception as e:
                logger.warning(f'Backfill: failed to parse CLIP result {fpath}: {e}')

    return reports


def backfill(output_dir: str) -> None:
    """Scan existing output directories and populate the metadata DB.

    Safe to run multiple times (uses INSERT OR REPLACE).
    Skips directories already present in the DB to avoid redundant work.
    """
    if not os.path.isdir(output_dir):
        return

    conn = _get_conn()

    # --- Backfill eval reports ---
    try:
        from evalscope.utils.data_utils import load_single_report, scan_for_report_folders
        raw_reports = scan_for_report_folders(output_dir)
        eval_count = 0
        eval_skipped = 0
        # Pre-fetch existing task IDs that already have eval_backend set
        existing_done = {
            r[0] for r in conn.execute("SELECT task_id FROM eval_reports WHERE eval_backend != ''").fetchall()
        }
        for rn in raw_reports:
            try:
                # Extract task_id (prefix) from composite report_name
                from evalscope.utils.data_utils import process_report_name
                prefix, _, _ = process_report_name(rn)
                if prefix in existing_done:
                    eval_skipped += 1
                    continue
                report_list, datasets, _ = load_single_report(output_dir, rn)
                if not report_list:
                    continue
                first = report_list[0]
                # Same policy as live eval (eval.py _execute_task): max of
                # positive per-metric counts, -1 (全量) only if all are <= 0.
                total_num = _compute_total_num(report_list)
                dataset_names: list[str] = []
                score_sum = 0.0
                dataset_scores: dict[str, float | None] = {}
                for r in report_list:
                    dataset_names.append(r.dataset_name)
                    score_sum += r.score
                    score = r.score
                    if score is not None and score > 1:
                        score = score / 100
                    dataset_scores[r.dataset_name] = round(score, 4) if score is not None else None
                avg_score = round(score_sum / len(report_list), 4) if report_list else 0.0

                # Extract timestamp from directory name
                from evalscope.utils.data_utils import process_report_name
                prefix, _, _ = process_report_name(rn)
                timestamp = ''
                for fmt in ('%Y%m%d_%H%M%S', '%Y%m%d'):
                    try:
                        dt = datetime.strptime(prefix, fmt)
                        timestamp = dt.isoformat()
                        break
                    except ValueError:
                        continue
                if not timestamp:
                    dir_path = os.path.join(output_dir, prefix)
                    if os.path.isdir(dir_path):
                        mtime = os.path.getmtime(dir_path)
                        timestamp = datetime.fromtimestamp(mtime).isoformat()

                # Try to extract eval_backend from task config
                eval_backend = ''
                try:
                    import yaml as _yaml
                    config_path = os.path.join(output_dir, prefix, 'configs', 'task_config.yaml')
                    if os.path.isfile(config_path):
                        with open(config_path) as cf:
                            cfg = _yaml.safe_load(cf) or {}
                        eval_backend = cfg.get('eval_backend', '')
                except Exception:
                    pass

                upsert_eval_report(
                    task_id=prefix,
                    model_name=first.model_name,
                    dataset_name=', '.join(dataset_names) if len(dataset_names) > 1 else
                    (dataset_names[0] if dataset_names else ''),
                    score=avg_score,
                    num_samples=total_num,
                    timestamp=timestamp,
                    dataset_scores=dataset_scores,
                    eval_backend=eval_backend,
                    user_id=1,  # backfill: all existing data → admin
                )
                eval_count += 1
            except Exception as e:
                logger.debug(f'Backfill: skip eval report {rn}: {e}')
        if eval_count:
            logger.info(f'Backfill: indexed {eval_count} eval reports ({eval_skipped} already in DB)')
    except Exception as e:
        logger.warning(f'Backfill: eval reports failed: {e}')

    # --- Backfill AIGC / Audio reports from results.json ---
    try:
        aigc_audio_count = 0
        aigc_audio_skipped = 0
        existing_eval = {r[0] for r in conn.execute('SELECT task_id FROM eval_reports').fetchall()}
        for entry in sorted(os.listdir(output_dir)):
            if entry in existing_eval:
                continue
            task_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(task_dir):
                continue
            if upsert_aigc_audio_report(output_dir, entry):
                aigc_audio_count += 1
            elif os.path.isfile(os.path.join(task_dir, 'results.json')):
                aigc_audio_skipped += 1
        if aigc_audio_count:
            logger.info(f'Backfill: indexed {aigc_audio_count} AIGC/Audio reports ({aigc_audio_skipped} already in DB)')
    except Exception as e:
        logger.warning(f'Backfill: AIGC/Audio reports failed: {e}')

    # --- Backfill RAG (MTEB) reports from results/ directories ---
    # RAG tasks store MTEB JSON under <task_dir>/results/ (no reports/ dir),
    # so the generic scan_for_report_folders path above never indexes them.
    try:
        rag_count = 0
        rag_skipped = 0
        existing_rag = {r[0] for r in conn.execute("SELECT task_id FROM eval_reports").fetchall()}
        for entry in sorted(os.listdir(output_dir)):
            if entry in existing_rag:
                continue
            task_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(task_dir):
                continue
            results_dir = os.path.join(task_dir, 'results')
            if not os.path.isdir(results_dir):
                continue
            # Only index tasks that are actually RAG evals
            backend = ''
            try:
                import yaml as _yaml
                config_path = os.path.join(task_dir, 'configs', 'task_config.yaml')
                if os.path.isfile(config_path):
                    with open(config_path) as cf:
                        cfg = _yaml.safe_load(cf) or {}
                    backend = cfg.get('eval_backend', '')
            except Exception:
                pass
            if backend != 'RAGEval':
                continue
            report_list = _parse_mteb_results_dir(task_dir)
            if not report_list:
                rag_skipped += 1
                continue
            first = report_list[0]
            total_num = sum(r.num or 0 for r in report_list)
            dataset_names = [r.dataset_name for r in report_list]
            score_sum = sum(r.score for r in report_list if r.score is not None)
            avg_score = round(score_sum / len(report_list), 4) if report_list else 0.0
            dataset_scores = {}
            for r in report_list:
                score = r.score
                if score is not None and score > 1:
                    score = score / 100
                dataset_scores[r.dataset_name] = round(score, 4) if score is not None else None
            # Timestamp from directory name (eval_<ts>) or mtime
            timestamp = ''
            try:
                import re as _re
                m = _re.search(r'(\d{14})', entry)
                if m:
                    timestamp = datetime.strptime(m.group(1), '%Y%m%d%H%M%S').isoformat()
            except ValueError:
                pass
            if not timestamp:
                timestamp = datetime.fromtimestamp(os.path.getmtime(task_dir)).isoformat()
            upsert_eval_report(
                task_id=entry,
                model_name=first.model_name,
                dataset_name=', '.join(dataset_names) if len(dataset_names) > 1 else
                (dataset_names[0] if dataset_names else ''),
                score=avg_score,
                num_samples=total_num,
                timestamp=timestamp,
                dataset_scores=dataset_scores,
                eval_backend='RAGEval',
                user_id=1,  # backfill: all existing data → admin
            )
            rag_count += 1
        if rag_count:
            logger.info(f'Backfill: indexed {rag_count} RAG reports ({rag_skipped} skipped)')
    except Exception as e:
        logger.warning(f'Backfill: RAG reports failed: {e}')

    # --- Backfill CLIP Benchmark reports from <model_name>/*.json ---
    # CLIP tasks write results to <task_dir>/<model_name>/<dataset>_<task>.json
    # (no reports/, results.json or MTEB results/ dir), so none of the paths
    # above index them.  Detect via progress.json pipeline == clip_benchmark.
    try:
        clip_count = 0
        existing_clip = {r[0] for r in conn.execute('SELECT task_id FROM eval_reports').fetchall()}
        for entry in sorted(os.listdir(output_dir)):
            if entry in existing_clip:
                continue
            task_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(task_dir):
                continue
            pj_path = os.path.join(task_dir, 'progress.json')
            if not os.path.isfile(pj_path):
                continue
            try:
                with open(pj_path, encoding='utf-8') as f:
                    pj = json.load(f)
            except Exception:
                continue
            if pj.get('pipeline') != 'clip_benchmark' or pj.get('status') != 'completed':
                continue
            report_list = _parse_clip_results_dir(task_dir)
            if not report_list:
                continue
            first = report_list[0]
            total_num = sum(r.num or 0 for r in report_list)
            dataset_names = [r.dataset_name for r in report_list]
            score_sum = sum(r.score for r in report_list if r.score is not None)
            avg_score = round(score_sum / len(report_list), 4) if report_list else 0.0
            dataset_scores = {}
            for r in report_list:
                score = r.score
                if score is not None and score > 1:
                    score = score / 100
                dataset_scores[r.dataset_name] = round(score, 4) if score is not None else None
            # Timestamp from directory name (eval_<ts>) or mtime
            timestamp = ''
            try:
                import re as _re
                m = _re.search(r'(\d{14})', entry)
                if m:
                    timestamp = datetime.strptime(m.group(1), '%Y%m%d%H%M%S').isoformat()
            except ValueError:
                pass
            if not timestamp:
                timestamp = datetime.fromtimestamp(os.path.getmtime(task_dir)).isoformat()
            upsert_eval_report(
                task_id=entry,
                model_name=first.model_name,
                dataset_name=', '.join(dataset_names) if len(dataset_names) > 1 else
                (dataset_names[0] if dataset_names else ''),
                score=avg_score,
                num_samples=total_num,
                timestamp=timestamp,
                dataset_scores=dataset_scores,
                eval_backend='RAGEval',
                user_id=1,  # backfill: all existing data → admin
            )
            clip_count += 1
        if clip_count:
            logger.info(f'Backfill: indexed {clip_count} CLIP Benchmark reports')
    except Exception as e:
        logger.warning(f'Backfill: CLIP reports failed: {e}')

    # --- Backfill perf tasks ---
    try:
        perf_count = 0
        perf_skipped = 0
        existing_perf = {r[0] for r in conn.execute('SELECT task_id FROM perf_tasks').fetchall()}
        for entry in sorted(os.listdir(output_dir), reverse=True):
            task_dir = os.path.join(output_dir, entry)
            perf_dir = os.path.join(task_dir, 'perf')
            if not os.path.isdir(task_dir) or not os.path.isdir(perf_dir):
                continue

            # Skip entries already in DB
            if entry in existing_perf:
                perf_skipped += 1
                continue

            model = 'N/A'
            api_val = 'N/A'
            dataset = 'N/A'
            runs = 0
            has_report = os.path.exists(os.path.join(perf_dir, 'perf_report.html'))
            timestamp = ''

            # Read args from first run subdirectory
            try:
                for search_dir in [task_dir, perf_dir]:
                    if not os.path.isdir(search_dir):
                        continue
                    for sub in sorted(os.listdir(search_dir)):
                        sub_dir = os.path.join(search_dir, sub)
                        if not os.path.isdir(sub_dir) or sub == 'perf':
                            continue
                        args_file = os.path.join(sub_dir, 'benchmark_args.json')
                        if os.path.isfile(args_file):
                            with open(args_file, 'r') as f:
                                args_data = json.load(f)
                            model = args_data.get('model', 'N/A')
                            api_val = args_data.get('api', 'N/A')
                            dataset = args_data.get('dataset_label') or args_data.get('dataset', 'N/A')
                            break
                    if model != 'N/A':
                        break
            except Exception:
                pass

            # Count runs
            try:
                for search_dir in [task_dir, perf_dir]:
                    if os.path.isdir(search_dir):
                        runs += sum(
                            1 for s in os.listdir(search_dir)
                            if os.path.isdir(os.path.join(search_dir, s)) and s != 'perf'
                        )
            except Exception:
                pass

            # Timestamp
            try:
                mtime = os.path.getmtime(task_dir)
                timestamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

            upsert_perf_task(
                task_id=entry,
                model=model,
                api=api_val,
                dataset=dataset,
                runs=runs,
                has_report=has_report,
                timestamp=timestamp,
                user_id=1,  # backfill: all existing data → admin
            )
            perf_count += 1
        if perf_count:
            logger.info(f'Backfill: indexed {perf_count} perf tasks ({perf_skipped} already in DB)')
    except Exception as e:
        logger.warning(f'Backfill: perf tasks failed: {e}')

    total_eval = conn.execute('SELECT COUNT(*) FROM eval_reports').fetchone()[0]
    total_perf = conn.execute('SELECT COUNT(*) FROM perf_tasks').fetchone()[0]
    logger.info(f'Backfill complete: {total_eval} eval reports, {total_perf} perf tasks in DB')

    # Force WAL checkpoint after bulk backfill writes
    checkpoint_db()

    # Update query planner statistics after bulk insert
    conn.execute('PRAGMA optimize')
    logger.debug('DB statistics optimized')


# --------------------------------------------------------------------------- #
# Compare reports CRUD                                                        #
# --------------------------------------------------------------------------- #

def save_compare_report(name: str, task_ids_json: str, task_count: int, backend: str = 'Perf', root_path: str = '', user_id: int = 1) -> int:
    """Save a compare report and return its ID."""
    from datetime import datetime

    with _write_lock:
        conn = _get_conn()
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            'INSERT INTO compare_reports (name, task_ids, created_at, task_count, backend, root_path, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (name, task_ids_json, created_at, task_count, backend, root_path, user_id),
        )
        conn.commit()
        return conn.execute('SELECT last_insert_rowid()').fetchone()[0]


def list_compare_reports(user_id: int = 1) -> list[dict]:
    """Return all saved compare reports for the given user, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        'SELECT id, name, task_ids, created_at, task_count, backend, root_path FROM compare_reports WHERE user_id = ? ORDER BY created_at DESC',
        (user_id,)
    ).fetchall()
    return [
        {
            'id': r[0],
            'name': r[1],
            'task_ids': r[2],
            'created_at': r[3],
            'task_count': r[4],
            'backend': r[5] or 'Perf',
            'root_path': r[6] or '',
        }
        for r in rows
    ]


def delete_compare_report(report_id: int, user_id: int | None = None) -> bool:
    """Delete a compare report by ID. Returns True if deleted."""
    conn = _get_conn()
    if user_id is not None:
        cur = conn.execute('DELETE FROM compare_reports WHERE id = ? AND user_id = ?', (report_id, user_id))
    else:
        cur = conn.execute('DELETE FROM compare_reports WHERE id = ?', (report_id,))
    conn.commit()
    return cur.rowcount > 0
