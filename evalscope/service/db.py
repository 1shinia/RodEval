"""SQLite metadata store for eval reports and perf tasks.

Provides fast listing/filtering without scanning the filesystem on every
request.  The database file lives at ``{OUTPUT_DIR}/evalscope_meta.db``.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from evalscope.utils.logger import get_logger

from .time_utils import (epoch_to_utc_iso, legacy_datetime_to_utc_iso, normalize_persisted_timestamp, utc_now_iso)

logger = get_logger()

_local = threading.local()
_db_path: str | None = None

# Process-wide write lock. All SQLite write operations (upserts, deletes,
# task-state mutations) go through this lock so that concurrent request
# threads (waitress workers) cannot deadlock on the database write lock
# when multiple tasks complete at the same time. Reads are lock-free.
_write_lock = threading.Lock()


def _write(fn, *, retries: int = 5, backoff: float = 0.2) -> Any:
    """Serialise *fn* (a callable taking the thread-local connection) under
    ``_write_lock`` with locked-error retry and rollback-on-failure.

    This is the ONLY sanctioned way to mutate the database.  Every write
    helper below routes through it so that lock discipline, retry policy and
    half-open-transaction cleanup stay in one place.
    """
    with _write_lock:
        conn = _get_conn()
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                result = fn(conn)
                conn.commit()
                return result
            except sqlite3.OperationalError as e:
                last_err = e
                # Roll back any half-open implicit transaction so the
                # thread-local connection isn't left holding a stale write
                # intent after a busy/lock failure.
                try:
                    conn.rollback()
                except Exception:
                    pass
                if 'locked' in str(e) and attempt < retries - 1:
                    import time
                    time.sleep(backoff * (attempt + 1))
                    continue
                raise
            except Exception:
                # Non-lock errors: still clean up a possible open transaction
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
        assert last_err is not None
        raise last_err

# ---------------------------------------------------------------------------
# Schema versioning — simple linear migration system
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 15  # Bump when adding migrations below

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
    # NOTE (2026-08, history drift): this migration entry was originally
    # "add COLLATE NOCASE index for model_name" (applied 2026-07-03). Its SQL
    # was later REWRITTEN IN PLACE to add eval_backend instead, and the column
    # was manually ALTERed onto the then-live DB — producing a "ghost column"
    # with no migration record, and a NOCASE index with a mismatched record.
    # Historical DBs keep both; fresh DBs lost the index (restored by v9).
    # Released migrations are IMMUTABLE — append-only from here on (guarded by
    # tests/test_db_migrations.py fingerprint test).
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
    (
        9, 'ensure model_name NOCASE index', '''
        -- Reconciles schema drift with historical DBs: IF NOT EXISTS is a
        -- no-op (silent SKIP) on DBs that already carry this index from the
        -- pre-drift v3, and creates it on fresh DBs. Definition must stay in
        -- sync with the original: model_name COLLATE NOCASE.
        CREATE INDEX IF NOT EXISTS idx_eval_reports_model_nocase
            ON eval_reports(model_name COLLATE NOCASE);
    '''
    ),
    (
        10, 'add has_errors/error_note to eval_reports', '''
        ALTER TABLE eval_reports ADD COLUMN has_errors INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE eval_reports ADD COLUMN error_note TEXT NOT NULL DEFAULT '';
        UPDATE eval_reports SET has_errors = 0 WHERE has_errors IS NULL;
        UPDATE eval_reports SET error_note = '' WHERE error_note IS NULL;
    '''
    ),
    (
        11, 'add status CHECK to task_state', '''
        -- SQLite cannot ALTER a CHECK constraint, so rebuild the runtime
        -- table. Normalize any out-of-spec status first so the copy cannot
        -- fail on the new constraint.
        UPDATE task_state SET status = 'failed'
            WHERE status NOT IN ('running', 'completed', 'failed', 'stopped', 'orphaned');
        CREATE TABLE task_state_new (
            task_id    TEXT PRIMARY KEY,
            task_type  TEXT NOT NULL,
            status     TEXT NOT NULL DEFAULT 'running'
                       CHECK (status IN ('running', 'completed', 'failed', 'stopped', 'orphaned')),
            pid        INTEGER,
            model      TEXT DEFAULT '',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO task_state_new (task_id, task_type, status, pid, model, started_at, updated_at)
            SELECT task_id, task_type, status, pid, model, started_at, updated_at FROM task_state;
        DROP TABLE task_state;
        ALTER TABLE task_state_new RENAME TO task_state;
        CREATE INDEX IF NOT EXISTS idx_task_state_status ON task_state(status);
        CREATE INDEX IF NOT EXISTS idx_task_state_task_type ON task_state(task_type);
    '''
    ),
    (
        12, 'add user_id to task_state', '''
        ALTER TABLE task_state ADD COLUMN user_id INTEGER DEFAULT 0;
        UPDATE task_state SET user_id = 0 WHERE user_id IS NULL;
    '''
    ),
    (
        13, 'add tenant-aware composite indexes', '''
        CREATE INDEX IF NOT EXISTS idx_eval_reports_user_timestamp
            ON eval_reports(user_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_eval_reports_user_backend_timestamp
            ON eval_reports(user_id, eval_backend, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_eval_reports_user_model_nocase_timestamp
            ON eval_reports(user_id, model_name COLLATE NOCASE, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_perf_tasks_user_timestamp
            ON perf_tasks(user_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_perf_tasks_user_model_timestamp
            ON perf_tasks(user_id, model, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_compare_reports_user_created
            ON compare_reports(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_task_state_user_status
            ON task_state(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_token_blacklist_expires
            ON token_blacklist(expires_at);
    '''
    ),
    (
        14, 'add soft-delete marker to users', '''
        ALTER TABLE users ADD COLUMN deleted_at TEXT DEFAULT NULL;
        CREATE INDEX IF NOT EXISTS idx_users_active_role
            ON users(deleted_at, role);
    '''
    ),
    (
        15, 'normalize metadata timestamps to UTC', '''
        -- The data rewrite is implemented by _migrate_v15_timestamps_to_utc
        -- so offset-aware values can be preserved and legacy naive values can
        -- be interpreted using EVALSCOPE_LEGACY_UTC_OFFSET_HOURS.
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


def _iter_sql_statements(sql: str):
    """Yield complete SQL statements without using ``executescript``.

    ``sqlite3.Connection.executescript`` may commit before running the script,
    which makes a multi-statement migration vulnerable to partial application.
    Feeding complete statements through ``execute`` keeps every migration
    inside the explicit transaction started by :func:`_migrate`.
    """
    buf = ''
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            statement = buf.strip()
            buf = ''
            if statement:
                yield statement
    if buf.strip():
        raise sqlite3.OperationalError('Incomplete SQL statement in migration')


def _repair_partial_v11(conn: sqlite3.Connection) -> None:
    """Recover shapes left by the legacy non-atomic v11 migration.

    Before migrations became transactional, an interrupted table rebuild could
    leave both ``task_state`` and ``task_state_new`` (failure before DROP), or
    only ``task_state_new`` (failure after DROP but before RENAME). Normalize
    either state so the immutable v11 script can be replayed safely.
    """
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('task_state', 'task_state_new')"
        ).fetchall()
    }
    if 'task_state_new' not in tables:
        return
    if 'task_state' in tables:
        conn.execute('DROP TABLE task_state_new')
    else:
        conn.execute('ALTER TABLE task_state_new RENAME TO task_state')
    conn.commit()
    logger.warning('Recovered interrupted legacy v11 task_state rebuild before retrying migration')


_V15_TIMESTAMP_COLUMNS: tuple[tuple[str, str], ...] = (
    ('eval_reports', 'timestamp'),
    ('perf_tasks', 'timestamp'),
    ('compare_reports', 'created_at'),
    ('task_state', 'started_at'),
    ('task_state', 'updated_at'),
    ('schema_version', 'applied_at'),
)


def _migrate_v15_timestamps_to_utc(conn: sqlite3.Connection) -> None:
    """Convert pre-v15 naive metadata timestamps to timezone-aware UTC.

    Historically the service wrote server-local naive strings.  The deployed
    environment for those rows was Asia/Shanghai (UTC+08:00).  The conversion
    offset is configurable via ``EVALSCOPE_LEGACY_UTC_OFFSET_HOURS`` so a fork
    that historically ran elsewhere can set the correct value before its first
    v15 startup.

    Already offset-aware values are converted by instant, not shifted again;
    empty or unparseable strings are intentionally preserved.
    """
    converted = 0
    for table, column in _V15_TIMESTAMP_COLUMNS:
        columns = {r[1] for r in conn.execute(f'PRAGMA table_info({table})').fetchall()}
        if column not in columns:
            continue
        rows = conn.execute(
            f'SELECT rowid, {column} FROM {table} '
            f"WHERE {column} IS NOT NULL AND TRIM({column}) != ''"
        ).fetchall()
        updates: list[tuple[str, int]] = []
        for rowid, value in rows:
            normalized = normalize_persisted_timestamp(value)
            if normalized is not None and normalized != value:
                updates.append((normalized, rowid))
        if updates:
            conn.executemany(f'UPDATE {table} SET {column} = ? WHERE rowid = ?', updates)
            converted += len(updates)
    logger.info(f'DB migration v15: normalized {converted} legacy timestamp value(s) to UTC')


_PYTHON_MIGRATIONS = {15: _migrate_v15_timestamps_to_utc}

# v11 rebuilds a table; v15 rewrites historical timestamps.  Both should have
# a rollback snapshot when upgrading an existing database.
_DESTRUCTIVE_MIGRATIONS = {11, 15}


def _migrate(conn: sqlite3.Connection, pre_migration_backup=None) -> None:
    """Apply pending migrations atomically up to :data:`SCHEMA_VERSION`.

    Every migration version is executed in its own ``BEGIN IMMEDIATE``
    transaction, including the schema_version insert.  A failure therefore
    rolls the whole version back instead of leaving a half-applied schema.

    Destructive migrations require a successful pre-migration backup when
    upgrading an existing database. Fresh databases do not need a rollback
    snapshot because they contain no user data yet.
    """
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

    pending = [m for m in _MIGRATIONS if current < m[0] <= SCHEMA_VERSION]
    needs_destructive_backup = current > 0 and any(v in _DESTRUCTIVE_MIGRATIONS for v, _, _ in pending)

    if pre_migration_backup is not None and current > 0:
        try:
            backup_path = pre_migration_backup()
        except Exception as e:
            if needs_destructive_backup:
                raise RuntimeError(f'Pre-migration backup failed before destructive migration: {e}') from e
            logger.warning(f'Pre-migration backup failed (non-fatal): {e}')
        else:
            if needs_destructive_backup and not backup_path:
                raise RuntimeError('Pre-migration backup did not produce a snapshot before destructive migration')
    elif needs_destructive_backup:
        raise RuntimeError('Destructive migration requires a pre-migration backup callback')

    if current < 11:
        _repair_partial_v11(conn)

    for version, description, sql in pending:
        logger.info(f'DB migration v{current}→v{version}: {description}')
        try:
            conn.execute('BEGIN IMMEDIATE')
            for statement in _iter_sql_statements(sql):
                conn.execute(statement)
            python_migration = _PYTHON_MIGRATIONS.get(version)
            if python_migration is not None:
                python_migration(conn)
            conn.execute(
                'INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)',
                (version, description, utc_now_iso()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        current = version

    logger.info(f'DB schema at v{current}')


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def _verify_schema() -> None:
    """Post-migration sanity checks for known schema-drift points.

    Two severity levels (NOT merged into one log line):
    - Missing ``eval_backend`` column  -> structural drift: every eval write
      (INSERT with explicit column names) would fail with OperationalError.
      Logged as ERROR with a remediation hint.
    - Missing / malformed NOCASE index -> query-plan degradation only (model
      filter ``= ? COLLATE NOCASE`` falls back to a full scan). Logged as
      WARNING.

    Never raises: startup must not be blocked by incidental drift, and the
    service stays available for diagnostics.
    """
    if _db_path is None:
        return
    conn = _get_conn()
    try:
        REQUIRED_COLUMNS = ('eval_backend', 'has_errors')
        cols = {r['name'] for r in conn.execute('PRAGMA table_info(eval_reports)')}
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            logger.error(
                f'Schema drift: eval_reports missing column(s): {", ".join(missing)}; '
                'eval writes will fail with OperationalError. Remediation: delete '
                'evalscope_meta.db and restart (backfill restores it), or run the '
                'corresponding ALTER TABLE ADD COLUMN statements.'
            )
        rows = conn.execute("PRAGMA index_xinfo('idx_eval_reports_model_nocase')").fetchall()
        if not rows:
            logger.warning(
                'Schema drift: idx_eval_reports_model_nocase index is missing; '
                'model filters fall back to a full scan. Migration v9 should have created it.'
            )
        elif not any(r['name'] == 'model_name' and r['coll'] == 'NOCASE' and r['key'] == 1 for r in rows):
            logger.warning(
                'Schema drift: idx_eval_reports_model_nocase exists but is malformed '
                '(expected model_name COLLATE NOCASE); model filters may not use it.'
            )
    except sqlite3.Error as e:
        logger.warning(f'Schema verification skipped (non-fatal): {e}')


def backup_db(output_dir: str | None = None, keep: int = 5, reason: str = 'startup') -> str | None:
    """Online backup of the meta database via SQLite's backup API (WAL-safe).

    Creates ``{output_dir}/backups/evalscope_meta_<reason>_<ts>.db`` and prunes
    old backups beyond *keep* per reason prefix.  Never raises: a failed
    backup must not block startup.
    """
    if _db_path is None:
        return None
    out_dir = output_dir or os.path.dirname(_db_path)
    try:
        if not os.path.isfile(_db_path):
            # sqlite3.connect would silently create an empty source DB and
            # "back it up" into a garbage snapshot — refuse instead.
            logger.warning(f'DB backup skipped ({reason}): source {_db_path} does not exist')
            return None
        os.makedirs(os.path.join(out_dir, 'backups'), exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        dest_path = os.path.join(out_dir, 'backups', f'evalscope_meta_{reason}_{ts}.db')

        src = sqlite3.connect(_db_path, timeout=30)
        try:
            dst = sqlite3.connect(dest_path)
            try:
                src.backup(dst)  # online: safe while service is running (WAL)
            finally:
                dst.close()
        finally:
            src.close()

        # Retention: keep newest *keep* files per reason prefix, by mtime.
        # The snapshot just written is included (newest mtime → survives).
        backups = sorted(
            (f for f in os.listdir(os.path.join(out_dir, 'backups'))
             if f.startswith(f'evalscope_meta_{reason}_') and f.endswith('.db')),
            key=lambda f: os.path.getmtime(os.path.join(out_dir, 'backups', f)),
        )
        for old in backups[:-keep]:
            try:
                os.remove(os.path.join(out_dir, 'backups', old))
            except OSError:
                pass
        logger.info(f'DB backup ({reason}): {dest_path}')
        return dest_path
    except Exception as e:
        logger.warning(f'DB backup skipped ({reason}, non-fatal): {e}')
        return None


def init_db(output_dir: str) -> None:
    """Initialise the database path and create tables if needed."""
    global _db_path
    _db_path = os.path.join(output_dir, 'evalscope_meta.db')
    os.makedirs(output_dir, exist_ok=True)
    conn = _get_conn()
    _migrate(conn, pre_migration_backup=lambda: backup_db(output_dir, reason='pre-migration'))
    _verify_schema()
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
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=30000')
        # Aggressive auto-checkpoint: flush WAL after ~800 KB instead of 4 MB default
        conn.execute('PRAGMA wal_autocheckpoint=200')
        _local.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Global task-id uniqueness / ownership helpers
# ---------------------------------------------------------------------------

_TASK_ID_TABLES = ('eval_reports', 'perf_tasks', 'task_state')


def task_id_exists(task_id: str) -> bool:
    """Return True when *task_id* is already known to any task table.

    Task directories are shared by eval and perf modules, therefore task IDs
    are treated as globally unique across task types rather than unique only
    within one table.
    """
    conn = _get_conn()
    for table in _TASK_ID_TABLES:
        if conn.execute(f'SELECT 1 FROM {table} WHERE task_id = ? LIMIT 1', (task_id,)).fetchone():
            return True
    return False


def new_task_id_available(output_dir: str, task_id: str) -> bool:
    """Return True only when a new task can safely claim *task_id*.

    Both metadata tables and the filesystem are checked. Reusing even an
    existing task owned by the same user is rejected because task writers use
    ``exist_ok=True`` and would otherwise overwrite historical artifacts.
    """
    if task_id_exists(task_id):
        return False
    return not os.path.exists(os.path.join(output_dir, task_id))


def task_ids_owned_by(table: str, task_ids: list[str], user_id: int) -> bool:
    """Return True iff every task id exists in *table* and belongs to user.

    ``table`` is restricted to known task tables so callers cannot inject SQL
    identifiers. Duplicate IDs are collapsed for the ownership check.
    """
    if table not in _TASK_ID_TABLES:
        raise ValueError(f'Unsupported task table: {table}')
    ids = list(dict.fromkeys(task_ids))
    if not ids:
        return False
    placeholders = ','.join('?' for _ in ids)
    conn = _get_conn()
    count = conn.execute(
        f'SELECT COUNT(*) FROM {table} WHERE user_id = ? AND task_id IN ({placeholders})',
        [user_id, *ids],
    ).fetchone()[0]
    return count == len(ids)


# ---------------------------------------------------------------------------
# Ownership markers — persist user_id next to the task directory so that
# backfill can restore attribution after meta.db loss (meta.db is NOT the
# only place ownership lives; see database-layer.md 三分法).
# Subprocess-side progress.json writers do full-file overwrites, so an owner
# field inside progress.json would be clobbered — hence a separate marker.


def write_owner_marker(task_dir: str, user_id: int) -> None:
    """Write ``<task_dir>/.owner`` without allowing ownership takeover.

    An existing marker may be rewritten by the same owner (resume/backfill),
    but a different user id is rejected. This is a second line of defence in
    case a caller forgets the new-task collision check.
    """
    try:
        os.makedirs(task_dir, exist_ok=True)
        owner_path = os.path.join(task_dir, '.owner')
        if os.path.isfile(owner_path):
            with open(owner_path) as f:
                existing = int(f.read().strip())
            if existing != int(user_id):
                raise PermissionError(
                    f'task directory already belongs to user {existing}, not {int(user_id)}'
                )
        tmp = os.path.join(task_dir, '.owner.tmp')
        with open(tmp, 'w') as f:
            f.write(str(int(user_id)))
        os.replace(tmp, owner_path)
    except PermissionError:
        raise
    except Exception as e:
        logger.debug(f'Owner marker write failed for {task_dir}: {e}')


def read_owner(task_dir: str) -> int:
    """Read the owner user id from ``<task_dir>/.owner``; 1 (=admin) if absent.

    Legacy directories created before markers existed fall back to 1,
    matching the historical backfill policy (all existing data → admin).
    """
    try:
        with open(os.path.join(task_dir, '.owner')) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 1


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
    has_errors: int = 0,
    error_note: str = '',
) -> None:
    def _op(conn: sqlite3.Connection) -> None:
        existing = conn.execute(
            'SELECT user_id FROM eval_reports WHERE task_id = ?', (task_id,)
        ).fetchone()
        if existing is not None and int(existing['user_id']) != int(user_id):
            raise PermissionError(
                f'task_id {task_id!r} is already owned by user {existing["user_id"]}'
            )
        conn.execute(
            '''INSERT INTO eval_reports
               (task_id, model_name, dataset_name, score, num_samples, timestamp,
                dataset_scores, eval_backend, user_id, has_errors, error_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                   model_name = excluded.model_name,
                   dataset_name = excluded.dataset_name,
                   score = excluded.score,
                   num_samples = excluded.num_samples,
                   timestamp = excluded.timestamp,
                   dataset_scores = excluded.dataset_scores,
                   eval_backend = excluded.eval_backend,
                   user_id = excluded.user_id,
                   has_errors = excluded.has_errors,
                   error_note = excluded.error_note''',
            (
                task_id, model_name, dataset_name, score, num_samples, timestamp,
                json.dumps(dataset_scores, ensure_ascii=False) if dataset_scores else None,
                eval_backend,
                user_id,
                has_errors, error_note,
            ),
        )

    _write(_op)


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

    # Available filter values (before filtering) — scoped to backend AND user
    # (filter dropdowns must not leak other tenants' model/dataset names)
    _backend_where = 'AND eval_backend = ?' if backend else ''
    _backend_params = [backend] if backend else []
    avail_models = [
        r[0] for r in conn.
        execute(f'SELECT DISTINCT model_name FROM eval_reports '
                f'WHERE model_name != "" AND user_id = ? {_backend_where} ORDER BY model_name',
                [user_id, *_backend_params]).fetchall()
    ]
    avail_datasets_raw = conn.execute(
        f'SELECT DISTINCT dataset_name FROM eval_reports '
        f'WHERE dataset_name != "" AND user_id = ? {_backend_where}',
        [user_id, *_backend_params],
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
                   timestamp, dataset_scores, has_errors, error_note
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
            'has_errors': row['has_errors'],
            'error_note': row['error_note'],
        })

    return items, total, avail_models, avail_datasets


def delete_eval_report(task_id: str, user_id: int | None = None) -> None:
    def _op(conn: sqlite3.Connection) -> None:
        if user_id is not None:
            conn.execute('DELETE FROM eval_reports WHERE task_id = ? AND user_id = ?', (task_id, user_id))
        else:
            conn.execute('DELETE FROM eval_reports WHERE task_id = ?', (task_id,))

    _write(_op)


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
    def _op(conn: sqlite3.Connection) -> None:
        existing = conn.execute(
            'SELECT user_id FROM perf_tasks WHERE task_id = ?', (task_id,)
        ).fetchone()
        if existing is not None and int(existing['user_id']) != int(user_id):
            raise PermissionError(
                f'task_id {task_id!r} is already owned by user {existing["user_id"]}'
            )
        conn.execute(
            '''INSERT INTO perf_tasks
               (task_id, model, api, dataset, runs, has_report, timestamp, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                   model = excluded.model,
                   api = excluded.api,
                   dataset = excluded.dataset,
                   runs = excluded.runs,
                   has_report = excluded.has_report,
                   timestamp = excluded.timestamp,
                   user_id = excluded.user_id''',
            (task_id, model, api, dataset, runs, int(has_report), timestamp, user_id),
        )

    _write(_op)


def cleanup_perf_tasks(output_dir: str, user_id: int | None = None) -> int:
    """Remove perf_tasks rows whose directories no longer exist on disk.

    Returns the number of rows removed.
    """

    def _op(conn: sqlite3.Connection) -> int:
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
        return len(stale)

    return _write(_op)


def cleanup_eval_reports(output_dir: str, user_id: int | None = None) -> int:
    """Remove eval_reports rows whose directories no longer exist on disk.

    Returns the number of rows removed.
    """

    def _op(conn: sqlite3.Connection) -> int:
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
        return len(stale)

    return _write(_op)


def _read_aigc_expected_metrics(output_dir: str, task_id: str) -> list[str]:
    """Expected AIGC metric names from the task config.

    Reads ``configs/task_config.yaml`` -> ``eval_config.eval.metrics``.
    Returns [] when the config is missing or unparsable (the caller then
    falls back to conservative "empty metrics dict" detection instead of
    guessing per-name — metrics like lpips are legitimate for img2img).
    """
    cfg_path = os.path.join(output_dir, task_id, 'configs', 'task_config.yaml')
    try:
        from evalscope.utils.io_utils import yaml_to_dict
        cfg = yaml_to_dict(cfg_path)
        ec = (cfg or {}).get('eval_config') or {}
        metrics = (ec.get('eval') or {}).get('metrics') or []
        if isinstance(metrics, list):
            return [str(m) for m in metrics]
    except Exception:
        pass
    return []


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

    # Determine primary score. Expected metrics come from the task config
    # (eval_config.eval.metrics); a missing expected metric silently yields a
    # fake 0.0 (indistinguishable from a genuine zero) — record it explicitly.
    # Metric-name mapping: config name -> results.json key.
    metric_key_map = {
        'clip_score': 'clip_score_mean',
        'fvd': 'fvd',
        'lpips': 'lpips_mean',
    }
    score = 0.0
    has_errors = 0
    error_note = ''
    if eval_backend == 'AIGC_EVAL':
        expected = _read_aigc_expected_metrics(output_dir, task_id)
        if not expected:
            # No config (e.g. legacy task): treat an empty metrics dict as
            # the anomaly signal instead of guessing per-name.
            if not metrics:
                has_errors, error_note = 1, 'no metrics produced'
        else:
            missing = [m for m in expected if metric_key_map.get(m) not in metrics]
            if missing:
                has_errors, error_note = 1, f'missing metric(s): {", ".join(missing)}'
        # Primary score: first expected metric with an actual value.
        for m in expected:
            key = metric_key_map.get(m)
            v = metrics.get(key) if key else None
            if v is not None:
                score = float(v)
                break
    elif tool == 'asr':
        wer = metrics.get('wer')
        score = (1.0 - wer) if wer is not None else 0.0
        if 'wer' not in metrics:
            has_errors, error_note = 1, 'missing wer metric'
    elif tool == 'tts':
        wer = metrics.get('wer_avg')
        score = (1.0 - wer) if wer is not None else 0.0
        if 'wer_avg' not in metrics:
            has_errors, error_note = 1, 'missing wer_avg metric'

    # Timestamp
    ts = data.get('timestamp')
    if ts:
        timestamp = epoch_to_utc_iso(float(ts))
    else:
        try:
            timestamp = epoch_to_utc_iso(os.path.getmtime(results_file))
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
        has_errors=has_errors,
        error_note=error_note,
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

    # Available filter values — scoped to user (no cross-tenant leakage)
    avail_models = [
        r[0] for r in conn.
        execute('SELECT DISTINCT model FROM perf_tasks '
                'WHERE model != "" AND model != "N/A" AND user_id = ? ORDER BY model',
                (user_id,)).fetchall()
    ]
    avail_datasets = [
        r[0] for r in conn.execute(
            'SELECT DISTINCT dataset FROM perf_tasks '
            'WHERE dataset != "" AND dataset != "N/A" AND user_id = ? ORDER BY dataset',
            (user_id,)
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
    def _op(conn: sqlite3.Connection) -> None:
        if user_id is not None:
            conn.execute('DELETE FROM perf_tasks WHERE task_id = ? AND user_id = ?', (task_id, user_id))
        else:
            conn.execute('DELETE FROM perf_tasks WHERE task_id = ?', (task_id,))

    _write(_op)


# ---------------------------------------------------------------------------
# Task state (running/completed/failed) — for persistence across restarts
# ---------------------------------------------------------------------------


def upsert_task_state(
    task_id: str,
    task_type: str,
    status: str,
    pid: int | None = None,
    model: str = '',
    user_id: int = 0,
) -> None:
    """Insert or update a task's runtime state.

    Status values: 'running', 'completed', 'failed', 'stopped', 'orphaned'.
    """

    def _op(conn: sqlite3.Connection) -> None:
        now = utc_now_iso()
        conn.execute(
            '''INSERT INTO task_state (task_id, task_type, status, pid, model, user_id, started_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                   status = excluded.status,
                   pid = excluded.pid,
                   updated_at = excluded.updated_at''',
            (task_id, task_type, status, pid, model, user_id, now, now),
        )

    _write(_op)


def delete_task_state(task_id: str) -> None:
    _write(lambda conn: conn.execute('DELETE FROM task_state WHERE task_id = ?', (task_id, )))


def _mark_orphaned_tasks(task_ids: list[str]) -> int:
    """Mark the given task_ids 'orphaned' (single locked write transaction)."""
    now = utc_now_iso()

    def _op(conn: sqlite3.Connection) -> int:
        conn.executemany(
            "UPDATE task_state SET status = 'orphaned', updated_at = ? WHERE task_id = ?",
            [(now, tid) for tid in task_ids],
        )
        return len(task_ids)

    return _write(_op)


def sweep_orphaned_tasks() -> list[dict]:
    """Maintenance pass: mark dead-PID running tasks orphaned.

    This is a WRITE operation and must not be called from read paths.
    Returns the rows still alive after the sweep.
    """
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, user_id, started_at, updated_at
           FROM task_state WHERE status = 'running'
           ORDER BY started_at DESC'''
    ).fetchall()
    dead = [r['task_id'] for r in rows if not (r['pid'] and _pid_alive(r['pid']))]
    if dead:
        for tid in dead:
            logger.info(f'Auto-orphaned zombie task {tid}')
        _mark_orphaned_tasks(dead)
    return [dict(r) for r in rows if r['task_id'] not in set(dead)]


def list_running_tasks() -> list[dict]:
    """Return all tasks with status='running' whose PID is still alive.

    Pure read: never mutates.  Zombie detection lives in
    :func:`sweep_orphaned_tasks` so queries stay side-effect free.
    """
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, user_id, started_at, updated_at
           FROM task_state WHERE status = 'running'
           ORDER BY started_at DESC'''
    ).fetchall()
    return [
        dict(r) for r in rows
        if r['pid'] and _pid_alive(r['pid'])
    ]


def get_all_task_states() -> list[dict]:
    """Return all task states (for debugging / admin)."""
    conn = _get_conn()
    rows = conn.execute(
        '''SELECT task_id, task_type, status, pid, model, user_id, started_at, updated_at
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
    if _db_path is None:
        return []

    pid_file = os.path.join(os.path.dirname(_db_path), 'evalscope_service.pid')
    old_pid = _read_service_pid(pid_file)

    # If old service is still alive, its tasks are legitimate — skip recovery.
    if old_pid is not None and _pid_alive(old_pid):
        logger.info(f'Previous service (PID {old_pid}) is still running — skipping stale task recovery.')
        return []

    conn = _get_conn()
    rows = conn.execute("SELECT task_id FROM task_state WHERE status = 'running'").fetchall()
    if not rows:
        return []

    orphaned = [row['task_id'] for row in rows]
    _mark_orphaned_tasks(orphaned)
    logger.info(f'Recovered {len(orphaned)} stale tasks from dead service (PID {old_pid}): {orphaned}')
    return orphaned


def cleanup_task_state(days: int = 7) -> int:
    """Remove stale 'orphaned'/'failed' task_state rows older than *days*.

    Rows accumulate because they are only deleted when the user explicitly
    removes a task; a dead subprocess (OOM/segfault) or a failed run leaves a
    row behind forever (current production DB had ~40 rows, oldest 2+ months).
    Called on startup AFTER recover_stale_tasks() so freshly-orphaned rows are
    still protected by the retention window. Retention matches the task-log
    cleanup policy (incomplete task dirs older than 7 days).

    Returns the number of rows removed.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def _op(conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            "DELETE FROM task_state WHERE status IN ('orphaned', 'failed') AND updated_at < ?",
            (cutoff,),
        )
        return cur.rowcount

    removed = _write(_op)
    if removed:
        logger.info(f'task_state cleanup: removed {removed} stale rows (older than {days}d)')
    return removed


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

                # Prefer mtime (epoch is timezone-independent); only parse a
                # legacy date-encoded directory name when mtime is unavailable.
                from evalscope.utils.data_utils import process_report_name
                prefix, _, _ = process_report_name(rn)
                timestamp = ''
                dir_path = os.path.join(output_dir, prefix)
                if os.path.isdir(dir_path):
                    try:
                        mtime = os.path.getmtime(dir_path)
                        timestamp = epoch_to_utc_iso(mtime)
                    except OSError:
                        pass
                if not timestamp:
                    for fmt in ('%Y%m%d_%H%M%S', '%Y%m%d'):
                        try:
                            dt = datetime.strptime(prefix, fmt)
                            timestamp = legacy_datetime_to_utc_iso(dt)
                            break
                        except ValueError:
                            continue

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
                    user_id=read_owner(os.path.join(output_dir, prefix)),  # .owner marker; legacy dirs → admin
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
            if upsert_aigc_audio_report(output_dir, entry, user_id=read_owner(task_dir)):
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
            # Prefer mtime; legacy 14-digit names are only a fallback.
            timestamp = ''
            try:
                timestamp = epoch_to_utc_iso(os.path.getmtime(task_dir))
            except OSError:
                pass
            if not timestamp:
                try:
                    import re as _re
                    m = _re.search(r'(\d{14})', entry)
                    if m:
                        timestamp = legacy_datetime_to_utc_iso(datetime.strptime(m.group(1), '%Y%m%d%H%M%S'))
                except ValueError:
                    pass
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
                user_id=read_owner(task_dir),  # .owner marker; legacy dirs → admin
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
            # Prefer mtime; legacy 14-digit names are only a fallback.
            timestamp = ''
            try:
                timestamp = epoch_to_utc_iso(os.path.getmtime(task_dir))
            except OSError:
                pass
            if not timestamp:
                try:
                    import re as _re
                    m = _re.search(r'(\d{14})', entry)
                    if m:
                        timestamp = legacy_datetime_to_utc_iso(datetime.strptime(m.group(1), '%Y%m%d%H%M%S'))
                except ValueError:
                    pass
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
                user_id=read_owner(task_dir),  # .owner marker; legacy dirs → admin
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
                timestamp = epoch_to_utc_iso(mtime)
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
                user_id=read_owner(task_dir),  # .owner marker; legacy dirs → admin
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

    created_at = utc_now_iso()

    def _op(conn: sqlite3.Connection) -> int:
        conn.execute(
            'INSERT INTO compare_reports (name, task_ids, created_at, task_count, backend, root_path, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (name, task_ids_json, created_at, task_count, backend, root_path, user_id),
        )
        return conn.execute('SELECT last_insert_rowid()').fetchone()[0]

    return _write(_op)


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
    def _op(conn: sqlite3.Connection) -> int:
        if user_id is not None:
            cur = conn.execute('DELETE FROM compare_reports WHERE id = ? AND user_id = ?', (report_id, user_id))
        else:
            cur = conn.execute('DELETE FROM compare_reports WHERE id = ?', (report_id,))
        return cur.rowcount

    return _write(_op) > 0
