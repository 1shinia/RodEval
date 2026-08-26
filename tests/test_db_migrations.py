"""Migration regression tests for evalscope.service.db.

Three guards, per the P2 schema-drift hardening (2026-08):

1. Fingerprint test — released migrations (v1–v8) are IMMUTABLE. Rewriting any
   of them after release must fail CI. Adding a new migration (v9+) requires
   appending its (version, description, sql) to RELEASED here.
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

import pytest

from evalscope.service import db


# --------------------------------------------------------------------------- #
# Released migration fingerprint (v1–v10). IMMUTABLE — do not edit.
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
    """The fingerprint must match db.py's first 8 migrations exactly.

    Any in-place rewrite of a released migration fails this test on purpose —
    released migrations are append-only.
    """
    assert len(RELEASED) == 12
    assert db._MIGRATIONS[: len(RELEASED)] == RELEASED
    assert db.SCHEMA_VERSION == len(db._MIGRATIONS) == 12


def test_fresh_db_converges(tmp_path):
    """A brand-new DB must reach the reconciled schema in one pass."""
    db.init_db(str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / 'evalscope_meta.db'))
    try:
        assert _schema_versions(conn) == list(range(1, db.SCHEMA_VERSION + 1))
        assert 'eval_backend' in _columns(conn, 'eval_reports')
        assert 'has_errors' in _columns(conn, 'eval_reports')
        assert 'error_note' in _columns(conn, 'eval_reports')
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