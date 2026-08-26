"""Tests for meta.db online backup (backup_db) and admin bootstrap (ensure_admin_user).

Covers the Phase-2 reliability work (2026-08-26):
- backup_db: WAL-safe online copy + per-reason retention pruning
- ensure_admin_user: preserves active admins and reactivates a soft-deleted
  default admin instead of colliding with the username UNIQUE constraint
"""
import os
import sqlite3

import pytest

from evalscope.service import db


@pytest.fixture(autouse=True)
def _reset_db_state():
    """Keep module-global DB state from leaking between tmp_path tests."""
    conn = getattr(db._local, 'conn', None)
    if conn is not None:
        conn.close()
    db._local.conn = None
    db._db_path = None
    yield
    conn = getattr(db._local, 'conn', None)
    if conn is not None:
        conn.close()
    db._local.conn = None
    db._db_path = None


def test_backup_db_creates_snapshot_and_prunes(tmp_path):
    db.init_db(str(tmp_path))
    bdir = tmp_path / 'backups'
    # Pre-existing backups beyond the keep=5 window
    bdir.mkdir()
    for i in range(7):
        (bdir / f'evalscope_meta_startup_old{i}.db').write_bytes(b'stale')

    dest = db.backup_db(str(tmp_path), keep=5, reason='startup')

    assert dest is not None and os.path.isfile(dest)
    # Snapshot must be a readable SQLite file with the schema_version table
    conn = sqlite3.connect(dest)
    try:
        versions = conn.execute('SELECT version FROM schema_version').fetchall()
    finally:
        conn.close()
    assert [v[0] for v in versions] == list(range(1, db.SCHEMA_VERSION + 1))

    # Retention: 7 stale + 1 fresh → newest 5 kept
    remaining = sorted(p.name for p in bdir.glob('evalscope_meta_startup_*.db'))
    assert len(remaining) == 5
    assert os.path.basename(dest) in remaining


def test_backup_db_never_raises_when_db_missing(tmp_path, monkeypatch):
    # _db_path pointing at a nonexistent file → warn + return None, no raise
    monkeypatch.setattr(db, '_db_path', str(tmp_path / 'nope.db'))
    assert db.backup_db(str(tmp_path), reason='startup') is None


def test_ensure_admin_user_create_once(tmp_path):
    from evalscope.service.blueprints.auth import ensure_admin_user

    db.init_db(str(tmp_path))
    conn = db._get_conn()
    conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at) "
        "VALUES ('keeper', 'x', 'admin', '2026-01-01')"
    )
    conn.commit()

    # Admin exists → bootstrap must NOT touch users table at all
    ensure_admin_user()
    rows = conn.execute("SELECT username FROM users WHERE role='admin'").fetchall()
    assert [r['username'] for r in rows] == ['keeper']

    # Fresh DB without any admin → exactly one 'admin' row created
    conn.execute("DELETE FROM users")
    conn.commit()
    ensure_admin_user()
    rows = conn.execute(
        "SELECT username, role FROM users WHERE username='admin'"
    ).fetchall()
    assert len(rows) == 1 and rows[0]['role'] == 'admin'


def test_ensure_admin_user_reactivates_soft_deleted_default(tmp_path, monkeypatch):
    from evalscope.service.blueprints.auth import ensure_admin_user

    monkeypatch.setenv('EVALSCOPE_ADMIN_PASSWORD', 'restored-secret')
    db.init_db(str(tmp_path))
    conn = db._get_conn()
    conn.execute(
        "INSERT INTO users (username, password_hash, role, created_at, deleted_at) "
        "VALUES ('admin', 'old-hash', 'admin', '2026-01-01T00:00:00+00:00', '2026-08-01T00:00:00+00:00')"
    )
    conn.commit()
    admin_id = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()['id']

    ensure_admin_user()

    row = conn.execute(
        "SELECT id, role, password_hash, deleted_at FROM users WHERE username='admin'"
    ).fetchone()
    assert row['id'] == admin_id
    assert row['role'] == 'admin'
    assert row['deleted_at'] is None
    assert row['password_hash'] != 'old-hash'
    assert conn.execute("SELECT COUNT(*) FROM users WHERE username='admin'").fetchone()[0] == 1
