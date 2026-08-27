"""Migration regression tests for evalscope.service.db.

Three guards, per the P2 schema-drift hardening (2026-08):

1. Fingerprint test — released migrations are IMMUTABLE. Rewriting any of
   them after release must fail CI. Adding a migration requires appending its
   (version, description, sql) tuple to RELEASED here.
2. Fresh-DB test — a brand-new DB runs the full migration chain and must end
   at SCHEMA_VERSION with the reconciled schema: eval_backend column +
   model_name NOCASE index + complete version history 1..SCHEMA_VERSION.
3. Historic-DB test — a DB built with the *pre-drift* history (v3 created the
   NOCASE index; eval_backend is a manually ALTERed ghost column with no
   migration record — the real production shape) must upgrade through v9,
   leave existing history untouched, and converge to the same schema.

Run (hermes env):  python -m pytest tests/test_db_migrations.py -q
"""
import sqlite3
import threading

import pytest

from evalscope.service import db


# --------------------------------------------------------------------------- #
# Released migration fingerprint (v1–v18). IMMUTABLE — do not edit.
# When appending a new migration to db.py, append it here too.
# --------------------------------------------------------------------------- #
RELEASED: list[tuple[int, str, str]] = [
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
    (
        16, 'add global task registry', '''
        CREATE TABLE IF NOT EXISTS task_registry (
            task_id     TEXT PRIMARY KEY,
            task_kind   TEXT NOT NULL CHECK (task_kind IN ('eval', 'perf')),
            user_id     INTEGER NOT NULL,
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_registry_user_kind_created
            ON task_registry(user_id, task_kind, created_at DESC);
    '''
    ),
    (
        17, 'normalize eval report datasets', '''
        CREATE TABLE IF NOT EXISTS eval_report_datasets (
            task_id       TEXT NOT NULL,
            user_id       INTEGER NOT NULL,
            dataset_name  TEXT NOT NULL COLLATE NOCASE,
            score         REAL,
            position      INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (task_id, dataset_name),
            FOREIGN KEY (task_id) REFERENCES eval_reports(task_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_eval_report_datasets_user_name_task
            ON eval_report_datasets(user_id, dataset_name COLLATE NOCASE, task_id);
        CREATE INDEX IF NOT EXISTS idx_eval_report_datasets_user_name_score
            ON eval_report_datasets(user_id, dataset_name COLLATE NOCASE, score);
        CREATE INDEX IF NOT EXISTS idx_eval_report_datasets_task_position
            ON eval_report_datasets(task_id, position);
    '''
    ),
    (
        18, 'add tenant-aware filter indexes', '''
        CREATE INDEX IF NOT EXISTS idx_eval_reports_user_score_timestamp
            ON eval_reports(user_id, score, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_perf_tasks_user_dataset_timestamp
            ON perf_tasks(user_id, dataset, timestamp DESC);
        CREATE TRIGGER IF NOT EXISTS trg_eval_reports_task_registry_insert
        BEFORE INSERT ON eval_reports
        BEGIN
            INSERT INTO task_registry (task_id, task_kind, user_id, created_at)
            VALUES (
                NEW.task_id, 'eval', COALESCE(NEW.user_id, 1),
                COALESCE(NULLIF(NEW.timestamp, ''), datetime('now'))
            )
            ON CONFLICT(task_id) DO NOTHING;
            SELECT CASE WHEN EXISTS (
                SELECT 1 FROM task_registry
                WHERE task_id = NEW.task_id
                  AND (task_kind != 'eval' OR user_id != COALESCE(NEW.user_id, 1))
            ) THEN RAISE(ABORT, 'task registry conflict for eval report') END;
        END;
        CREATE TRIGGER IF NOT EXISTS trg_perf_tasks_task_registry_insert
        BEFORE INSERT ON perf_tasks
        BEGIN
            INSERT INTO task_registry (task_id, task_kind, user_id, created_at)
            VALUES (
                NEW.task_id, 'perf', COALESCE(NEW.user_id, 1),
                COALESCE(NULLIF(NEW.timestamp, ''), datetime('now'))
            )
            ON CONFLICT(task_id) DO NOTHING;
            SELECT CASE WHEN EXISTS (
                SELECT 1 FROM task_registry
                WHERE task_id = NEW.task_id
                  AND (task_kind != 'perf' OR user_id != COALESCE(NEW.user_id, 1))
            ) THEN RAISE(ABORT, 'task registry conflict for perf task') END;
        END;
        CREATE TRIGGER IF NOT EXISTS trg_task_state_task_registry_insert
        BEFORE INSERT ON task_state
        BEGIN
            INSERT INTO task_registry (task_id, task_kind, user_id, created_at)
            VALUES (
                NEW.task_id,
                CASE WHEN NEW.task_type = 'perf' THEN 'perf' ELSE 'eval' END,
                CASE WHEN COALESCE(NEW.user_id, 0) > 0 THEN NEW.user_id ELSE 1 END,
                COALESCE(NULLIF(NEW.started_at, ''), datetime('now'))
            )
            ON CONFLICT(task_id) DO NOTHING;
            SELECT CASE WHEN EXISTS (
                SELECT 1 FROM task_registry
                WHERE task_id = NEW.task_id
                  AND (
                      task_kind != CASE WHEN NEW.task_type = 'perf' THEN 'perf' ELSE 'eval' END
                      OR (
                          COALESCE(NEW.user_id, 0) > 0
                          AND user_id != NEW.user_id
                      )
                  )
            ) THEN RAISE(ABORT, 'task registry conflict for task state') END;
        END;
    '''
    ),
]

# Pre-drift migration history (what the production DB actually recorded):
# v3 created the NOCASE index; eval_backend was added by a manual ALTER that
# left no migration record (the "ghost column" this hardening fixes).
HISTORIC_RELEASED: list[tuple[int, str, str]] = (
    RELEASED[:2]
    + [
        (
            3, 'add COLLATE NOCASE index for model_name',
            'CREATE INDEX IF NOT EXISTS idx_eval_reports_model_nocase\n'
            '    ON eval_reports(model_name COLLATE NOCASE);',
        ),
    ]
    + RELEASED[3:8]
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _reset_db_state():
    """isolate the module-global _db_path / thread-local connection per test."""
    db._db_path = None
    db._local.conn = None
    yield
    db._db_path = None
    db._local.conn = None


def _schema_versions(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute('SELECT version FROM schema_version ORDER BY version')]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]


def _nocase_index_ok(conn: sqlite3.Connection) -> bool:
    """True when the NOCASE index exists with the exact expected definition.

    PRAGMA index_xinfo columns: seqno, cid, name, desc, coll, key.
    """
    rows = conn.execute("PRAGMA index_xinfo('idx_eval_reports_model_nocase')").fetchall()
    return any(r[2] == 'model_name' and r[4] == 'NOCASE' and r[5] == 1 for r in rows)


def _build_historic_db(db_path: str) -> None:
    """Create a DB with the pre-drift history (v1–v8) + ghost eval_backend
    column, mirroring the real production DB before v9 shipped."""
    conn = sqlite3.connect(db_path)
    # The version-tracking table is created by _migrate() itself, not by any
    # migration entry — mirror that so the historic DB is a faithful replica.
    conn.execute(
        '''CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at  TEXT NOT NULL
        )'''
    )
    for version, description, sql in HISTORIC_RELEASED:
        conn.executescript(sql)
        conn.execute(
            'INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)',
            (version, description, '2026-08-24T00:00:00.000000'),
        )
    # Ghost column: manually ALTERed, no migration record (the 2026-07 event)
    conn.execute("ALTER TABLE eval_reports ADD COLUMN eval_backend TEXT DEFAULT ''")
    conn.execute("UPDATE eval_reports SET eval_backend = '' WHERE eval_backend IS NULL")
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_released_migrations_immutable():
    """The fingerprint must match db.py's released migrations exactly.

    Any in-place rewrite of a released migration fails this test on purpose —
    released migrations are append-only.
    """
    assert len(RELEASED) == 18
    assert db._MIGRATIONS[: len(RELEASED)] == RELEASED
    assert db.SCHEMA_VERSION == len(db._MIGRATIONS) == 18


def test_fresh_db_converges(tmp_path):
    """A brand-new DB must reach the reconciled schema in one pass."""
    db.init_db(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / 'evalscope_meta.db'))
    try:
        assert _schema_versions(conn) == list(range(1, db.SCHEMA_VERSION + 1))
        assert 'eval_backend' in _columns(conn, 'eval_reports')
        assert 'has_errors' in _columns(conn, 'eval_reports')
        assert 'error_note' in _columns(conn, 'eval_reports')
        assert 'deleted_at' in _columns(conn, 'users')
        assert {'task_id', 'task_kind', 'user_id', 'created_at'} <= set(_columns(conn, 'task_registry'))
        assert {'task_id', 'dataset_name', 'score', 'position'} <= set(_columns(conn, 'eval_report_datasets'))
        idx = {r[1] for r in conn.execute("PRAGMA index_list('eval_reports')").fetchall()}
        assert 'idx_eval_reports_user_timestamp' in idx
        assert 'idx_eval_reports_user_backend_timestamp' in idx
        assert 'idx_eval_reports_user_score_timestamp' in idx
        assert _nocase_index_ok(conn), 'NOCASE index missing/malformed on fresh DB'
    finally:
        conn.close()


def test_fresh_db_verify_schema_clean(tmp_path, caplog):
    """Startup self-check must pass without any drift warning/error."""
    db.init_db(str(tmp_path))
    assert not any('Schema drift' in r.message for r in caplog.records)


def test_historic_db_upgrade_v9(tmp_path):
    """A pre-drift historical DB (the real production shape) must upgrade
    through v9 without touching its history, keeping column + index intact."""
    db_path = str(tmp_path / 'evalscope_meta.db')
    _build_historic_db(db_path)

    db.init_db(str(tmp_path))  # applies v9 + runs _verify_schema

    conn = sqlite3.connect(db_path)
    try:
        versions = _schema_versions(conn)
        assert versions == list(range(1, db.SCHEMA_VERSION + 1))
        # History records are untouched: v3 keeps its ORIGINAL pre-drift text
        desc3 = conn.execute('SELECT description FROM schema_version WHERE version = 3').fetchone()[0]
        assert desc3 == 'add COLLATE NOCASE index for model_name'
        assert versions.count(9) == 1 and versions.count(3) == 1
        # Reconciled schema intact
        assert 'eval_backend' in _columns(conn, 'eval_reports')
        assert 'has_errors' in _columns(conn, 'eval_reports')
        assert 'error_note' in _columns(conn, 'eval_reports')
        assert _nocase_index_ok(conn), 'NOCASE index lost during v9 upgrade'
    finally:
        conn.close()


def test_historic_db_verify_schema_clean(tmp_path, caplog):
    """Self-check after upgrading a historic DB must be silent (no drift)."""
    db_path = str(tmp_path / 'evalscope_meta.db')
    _build_historic_db(db_path)
    db.init_db(str(tmp_path))
    assert not any('Schema drift' in r.message for r in caplog.records)

def test_migration_version_is_atomic(tmp_path, monkeypatch):
    """A failing multi-statement migration must leave no partial DDL behind."""
    baseline = db.SCHEMA_VERSION
    probe_version = baseline + 1
    db_path = str(tmp_path / 'atomic.db')
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    conn.execute("CREATE TABLE demo (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version VALUES (?, 'baseline', 'x')", (baseline,))
    conn.commit()

    monkeypatch.setattr(db, 'SCHEMA_VERSION', probe_version)
    monkeypatch.setattr(
        db, '_MIGRATIONS',
        [*db._MIGRATIONS, (probe_version, 'atomic failure probe',
         'ALTER TABLE demo ADD COLUMN should_rollback TEXT;\n'
         'INSERT INTO table_that_does_not_exist VALUES (1);')],
    )
    with pytest.raises(sqlite3.OperationalError):
        db._migrate(conn)

    assert 'should_rollback' not in _columns(conn, 'demo')
    assert _schema_versions(conn)[-1] == baseline
    conn.close()


def test_concurrent_migrators_converge_without_replaying_versions(tmp_path):
    """Two service processes starting together must converge on one history."""
    db_path = str(tmp_path / 'concurrent.db')
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _worker() -> None:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            barrier.wait(timeout=5)
            db._migrate(conn, pre_migration_backup=lambda: str(tmp_path / 'snapshot.db'))
        except Exception as e:
            errors.append(e)
        finally:
            conn.close()

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not errors
    conn = sqlite3.connect(db_path)
    try:
        assert _schema_versions(conn) == list(range(1, db.SCHEMA_VERSION + 1))
        assert conn.execute('SELECT COUNT(*) FROM schema_version').fetchone()[0] == db.SCHEMA_VERSION
    finally:
        conn.close()


def test_v15_converts_legacy_naive_timestamps_from_cst_to_utc(tmp_path, monkeypatch):
    """Pre-v15 naive metadata is interpreted as UTC+8 and normalized once."""
    monkeypatch.setenv('EVALSCOPE_LEGACY_UTC_OFFSET_HOURS', '8')
    monkeypatch.setattr(db, 'SCHEMA_VERSION', 14)
    db.init_db(str(tmp_path))

    # Close the thread-local connection before mutating with an independent
    # handle and before re-running init_db at the next schema version.
    db._local.conn.close()
    db._local.conn = None
    db_path = str(tmp_path / 'evalscope_meta.db')
    conn = sqlite3.connect(db_path)
    conn.execute(
        '''INSERT INTO eval_reports
           (task_id, model_name, dataset_name, timestamp, user_id, has_errors, error_note)
           VALUES ('eval_tz', 'm', 'd', '2026-08-26T16:00:00', 1, 0, '')'''
    )
    conn.execute(
        '''INSERT INTO perf_tasks
           (task_id, model, timestamp, user_id)
           VALUES ('perf_tz', 'm', '2026-08-26 16:00:00', 1)'''
    )
    conn.execute(
        '''INSERT INTO compare_reports
           (name, task_ids, created_at, task_count, backend, root_path, user_id)
           VALUES ('c', '[]', '2026-08-26T16:00:00', 0, 'Perf', '', 1)'''
    )
    conn.execute(
        '''INSERT INTO task_state
           (task_id, task_type, status, pid, model, started_at, updated_at, user_id)
           VALUES ('run_tz', 'eval', 'running', NULL, 'm',
                   '2026-08-26T16:00:00', '2026-08-26T16:30:00', 1)'''
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, 'SCHEMA_VERSION', 15)
    db.init_db(str(tmp_path))
    db._local.conn.close()
    db._local.conn = None

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT timestamp FROM eval_reports WHERE task_id='eval_tz'"
        ).fetchone()[0] == '2026-08-26T08:00:00+00:00'
        assert conn.execute(
            "SELECT timestamp FROM perf_tasks WHERE task_id='perf_tz'"
        ).fetchone()[0] == '2026-08-26T08:00:00+00:00'
        assert conn.execute(
            "SELECT created_at FROM compare_reports WHERE name='c'"
        ).fetchone()[0] == '2026-08-26T08:00:00+00:00'
        started, updated = conn.execute(
            "SELECT started_at, updated_at FROM task_state WHERE task_id='run_tz'"
        ).fetchone()
        assert started == '2026-08-26T08:00:00+00:00'
        assert updated == '2026-08-26T08:30:00+00:00'
    finally:
        conn.close()


def test_future_schema_version_is_rejected(tmp_path):
    """An older binary must never open a DB created by a newer schema."""
    db_path = str(tmp_path / 'evalscope_meta.db')
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE schema_version '
        '(version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)'
    )
    conn.execute(
        'INSERT INTO schema_version VALUES (?, ?, ?)',
        (db.SCHEMA_VERSION + 1, 'future', '2099-01-01T00:00:00+00:00'),
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match='newer than this application supports'):
        db.init_db(str(tmp_path))


def test_current_version_structural_drift_is_fatal(tmp_path):
    """A version marker alone cannot make a structurally broken DB healthy."""
    db_path = str(tmp_path / 'evalscope_meta.db')
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE schema_version '
        '(version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)'
    )
    conn.execute(
        'INSERT INTO schema_version VALUES (?, ?, ?)',
        (db.SCHEMA_VERSION, 'pretend-current', '2026-08-26T00:00:00+00:00'),
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match='Schema drift:'):
        db.init_db(str(tmp_path))
