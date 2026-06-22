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
# Connection management
# ---------------------------------------------------------------------------


def init_db(output_dir: str) -> None:
    """Initialise the database path and create tables if needed."""
    global _db_path
    _db_path = os.path.join(output_dir, 'evalscope_meta.db')
    os.makedirs(output_dir, exist_ok=True)
    conn = _get_conn()
    conn.executescript(
        '''
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
    '''
    )
    conn.commit()
    logger.info(f'SQLite metadata DB ready: {_db_path}')


def _get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection."""
    if _db_path is None:
        raise RuntimeError('init_db() has not been called')
    conn: sqlite3.Connection | None = getattr(_local, 'conn', None)
    if conn is None:
        conn = sqlite3.connect(_db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        _local.conn = conn
    return conn


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
        where.append('(LOWER(model_name) LIKE ? OR LOWER(dataset_name) LIKE ?)')
        params.extend([f'%{search}%', f'%{search}%'])
    if models:
        model_set = [m.strip().lower() for m in models.split(';') if m.strip()]
        if model_set:
            placeholders = ','.join('?' for _ in model_set)
            where.append(f'LOWER(model_name) IN ({placeholders})')
            params.extend(model_set)
    if datasets:
        ds_set = [d.strip().lower() for d in datasets.split(';') if d.strip()]
        if ds_set:
            ds_conds = []
            for ds in ds_set:
                ds_conds.append('LOWER(dataset_name) LIKE ?')
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
        where.append('(LOWER(model) LIKE ? OR LOWER(dataset) LIKE ?)')
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
    """Return all tasks with status='running'."""
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, started_at, updated_at
           FROM task_state WHERE status = 'running'
           ORDER BY started_at DESC'''
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_task_states() -> list[dict]:
    """Return all task states (for debugging / admin)."""
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, started_at, updated_at
           FROM task_state ORDER BY started_at DESC'''
    ).fetchall()
    return [dict(r) for r in rows]


def recover_stale_tasks() -> list[str]:
    """Mark 'running' tasks whose PID is dead as 'orphaned'.

    Called on server startup to clean up stale state from a previous crash.
    Returns the list of task_ids that were marked orphaned.
    """
    import signal
    conn = _get_conn()
    rows = conn.execute("SELECT task_id, pid FROM task_state WHERE status = 'running'").fetchall()

    orphaned: list[str] = []
    for row in rows:
        pid = row['pid']
        if pid is None:
            orphaned.append(row['task_id'])
            continue
        # Check if the process is still alive
        try:
            os.kill(pid, 0)  # Signal 0 = check existence, no actual signal sent
        except (OSError, ProcessLookupError):
            orphaned.append(row['task_id'])

    now = datetime.now().isoformat()
    for tid in orphaned:
        conn.execute(
            "UPDATE task_state SET status = 'orphaned', updated_at = ? WHERE task_id = ?",
            (now, tid),
        )
    if orphaned:
        conn.commit()
        logger.info(f'Recovered {len(orphaned)} stale tasks: {orphaned}')

    return orphaned


# ---------------------------------------------------------------------------
# Backfill — populate DB from existing filesystem data on first startup
# ---------------------------------------------------------------------------


def backfill(output_dir: str) -> None:
    """Scan existing output directories and populate the metadata DB.

    Safe to run multiple times (uses INSERT OR REPLACE).
    """
    if not os.path.isdir(output_dir):
        return

    conn = _get_conn()

    # --- Backfill eval reports ---
    try:
        from evalscope.utils.data_utils import load_single_report, scan_for_report_folders
        raw_reports = scan_for_report_folders(output_dir)
        eval_count = 0
        for rn in raw_reports:
            try:
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
            logger.info(f'Backfill: indexed {eval_count} eval reports')
    except Exception as e:
        logger.warning(f'Backfill: eval reports failed: {e}')

    # --- Backfill perf tasks ---
    try:
        perf_count = 0
        for entry in sorted(os.listdir(output_dir), reverse=True):
            task_dir = os.path.join(output_dir, entry)
            perf_dir = os.path.join(task_dir, 'perf')
            if not os.path.isdir(task_dir) or not os.path.isdir(perf_dir):
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
            logger.info(f'Backfill: indexed {perf_count} perf tasks')
    except Exception as e:
        logger.warning(f'Backfill: perf tasks failed: {e}')

    total_eval = conn.execute('SELECT COUNT(*) FROM eval_reports').fetchone()[0]
    total_perf = conn.execute('SELECT COUNT(*) FROM perf_tasks').fetchone()[0]
    logger.info(f'Backfill complete: {total_eval} eval reports, {total_perf} perf tasks in DB')
