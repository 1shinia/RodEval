"""Tests for meta.db online backup (backup_db) and admin bootstrap (ensure_admin_user).

Covers the Phase-2 reliability work (2026-08-26):
- backup_db: WAL-safe online copy + per-reason retention pruning
- ensure_admin_user: creates admin only when none exists, never resets
"""
import os
import sqlite3

from evalscope.service import db


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
