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

# ---------------------------------------------------------------------------
# Schema versioning — simple linear migration system
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2  # Bump when adding migrations below

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
    """
    conn = _get_conn()
    # TRUNCATE: reset the WAL file to zero bytes after checkpoint
    row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
    result = {'busy': row[0], 'log': row[1], 'checkpointed': row[2]}
    if result['checkpointed'] > 0:
        logger.debug(
            'WAL checkpoint: %d pages moved, %d in log, %d busy', result['checkpointed'], result['log'], result['busy']
        )
    return result


# ---------------------------------------------------------------------------
# Eval reports CRUD
# ---------------------------------------------------------------------------


def upsert_eval_report(
    task_id: str,
    model_name: str,
    dataset_name: str,
    score: float,
    num_samples: int,
    timestamp: str,
    dataset_scores: dict | None = None,
) -> None:
    conn = _get_conn()
    conn.execute(
        '''INSERT OR REPLACE INTO eval_reports
           (task_id, model_name, dataset_name, score, num_samples, timestamp, dataset_scores)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (
            task_id, model_name, dataset_name, score, num_samples, timestamp,
            json.dumps(dataset_scores, ensure_ascii=False) if dataset_scores else None
        ),
    )
    conn.commit()


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
) -> tuple[list[dict], int, list[str], list[str]]:
    """Return ``(items, total, available_models, available_datasets)``."""
    conn = _get_conn()
    where: list[str] = []
    params: list[Any] = []

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

    where_sql = f'WHERE {" AND ".join(where)}' if where else ''

    sort_map = {
        'score': 'score',
        'model': 'model_name',
        'dataset': 'dataset_name',
        'time': 'timestamp',
    }
    col = sort_map.get(sort_by, 'timestamp')
    direction = 'DESC' if sort_order == 'desc' else 'ASC'

    # Available filter values (before filtering)
    avail_models = [
        r[0] for r in conn.
        execute('SELECT DISTINCT model_name FROM eval_reports WHERE model_name != "" ORDER BY model_name').fetchall()
    ]
    avail_datasets_raw = conn.execute('SELECT DISTINCT dataset_name FROM eval_reports WHERE dataset_name != ""'
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


def delete_eval_report(task_id: str) -> None:
    conn = _get_conn()
    conn.execute('DELETE FROM eval_reports WHERE task_id = ?', (task_id, ))
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
) -> None:
    conn = _get_conn()
    conn.execute(
        '''INSERT OR REPLACE INTO perf_tasks
           (task_id, model, api, dataset, runs, has_report, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (task_id, model, api, dataset, runs, int(has_report), timestamp),
    )
    conn.commit()


def query_perf_tasks(
    search: str = '',
    filter_model: str = '',
    filter_dataset: str = '',
    sort_by: str = 'time',
    sort_order: str = 'desc',
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict], int, list[str], list[str]]:
    """Return ``(items, total, available_models, available_datasets)``."""
    conn = _get_conn()
    where: list[str] = []
    params: list[Any] = []

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


def delete_perf_task(task_id: str) -> None:
    conn = _get_conn()
    conn.execute('DELETE FROM perf_tasks WHERE task_id = ?', (task_id, ))
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
        # Pre-fetch existing task IDs to skip redundant processing
        existing_eval = {r[0] for r in conn.execute('SELECT task_id FROM eval_reports').fetchall()}
        for rn in raw_reports:
            try:
                # Extract task_id (prefix) from composite report_name
                from evalscope.utils.data_utils import process_report_name
                prefix, _, _ = process_report_name(rn)
                if prefix in existing_eval:
                    eval_skipped += 1
                    continue
                report_list, datasets, _ = load_single_report(output_dir, rn)
                if not report_list:
                    continue
                first = report_list[0]
                total_num = 0
                dataset_names: list[str] = []
                score_sum = 0.0
                dataset_scores: dict[str, float | None] = {}
                for r in report_list:
                    dataset_names.append(r.dataset_name)
                    total_num += r.num or 0
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

                upsert_eval_report(
                    task_id=prefix,
                    model_name=first.model_name,
                    dataset_name=', '.join(dataset_names) if len(dataset_names) > 1 else
                    (dataset_names[0] if dataset_names else ''),
                    score=avg_score,
                    num_samples=total_num,
                    timestamp=timestamp,
                    dataset_scores=dataset_scores,
                )
                eval_count += 1
            except Exception as e:
                logger.debug(f'Backfill: skip eval report {rn}: {e}')
        if eval_count:
            logger.info(f'Backfill: indexed {eval_count} eval reports ({eval_skipped} already in DB)')
    except Exception as e:
        logger.warning(f'Backfill: eval reports failed: {e}')

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
