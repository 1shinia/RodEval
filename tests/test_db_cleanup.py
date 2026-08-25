"""Regression tests for startup garbage-collection paths (P3 hardening).

Covers:
- cleanup_task_state(): stale orphaned/failed rows are removed after the
  retention window; fresh rows (including freshly-orphaned ones) survive.
- _cleanup_expired_tokens(): expired JWT blacklist entries are removed,
  unexpired ones survive.

Both run against an isolated temp DB — never the real outputs store.
"""
import sqlite3
import tempfile
from datetime import datetime, timedelta

import pytest

from evalscope.service import db as _db
from evalscope.service.blueprints.auth import _cleanup_expired_tokens


@pytest.fixture()
def iso_db():
    """Point the db module at a fresh temp DB per test, then reset."""
    tmp = tempfile.mkdtemp()
    _db._local.conn = None
    _db.init_db(tmp)
    yield tmp
    _db._db_path = None
    _db._local.conn = None


def _insert_task_state(conn: sqlite3.Connection, task_id: str, status: str, updated_at: str) -> None:
    conn.execute(
        '''INSERT INTO task_state (task_id, task_type, status, pid, model, started_at, updated_at)
           VALUES (?, 'eval', ?, NULL, '', ?, ?)''',
        (task_id, status, updated_at, updated_at),
    )


def test_cleanup_task_state_removes_only_stale(iso_db):
    conn = _db._get_conn()
    now = datetime.now()
    _insert_task_state(conn, 'stale_orphan', 'orphaned', (now - timedelta(days=30)).isoformat())
    _insert_task_state(conn, 'stale_failed', 'failed', (now - timedelta(days=14)).isoformat())
    _insert_task_state(conn, 'fresh_orphan', 'orphaned', (now - timedelta(days=1)).isoformat())
    _insert_task_state(conn, 'fresh_failed', 'failed', (now - timedelta(minutes=5)).isoformat())
    _insert_task_state(conn, 'running_keep', 'running', now.isoformat())  # never touched
    conn.commit()

    removed = _db.cleanup_task_state(days=7)

    assert removed == 2  # only stale_orphan + stale_failed
    remaining = {r[0] for r in conn.execute('SELECT task_id FROM task_state')}
    assert remaining == {'fresh_orphan', 'fresh_failed', 'running_keep'}


def test_cleanup_task_state_default_days(iso_db):
    conn = _db._get_conn()
    now = datetime.now()
    _insert_task_state(conn, 'old_orphan', 'orphaned', (now - timedelta(days=8)).isoformat())
    _insert_task_state(conn, 'new_orphan', 'orphaned', (now - timedelta(days=6)).isoformat())
    conn.commit()

    removed = _db.cleanup_task_state()  # default days=7
    assert removed == 1
    remaining = {r[0] for r in conn.execute('SELECT task_id FROM task_state')}
    assert remaining == {'new_orphan'}


def test_cleanup_expired_tokens_removes_only_expired(iso_db):
    conn = _db._get_conn()
    now = datetime.utcnow()
    expired_old = (now - timedelta(days=3)).isoformat()
    expired_recent = (now - timedelta(hours=1)).isoformat()
    unexpired = (now + timedelta(hours=48)).isoformat()
    rows = [
        ('jti_expired_old', expired_old),
        ('jti_expired_recent', expired_recent),
        ('jti_valid', unexpired),
    ]
    conn.executemany('INSERT INTO token_blacklist (jti, expires_at) VALUES (?, ?)', rows)
    conn.commit()

    _cleanup_expired_tokens()

    remaining = {r[0] for r in conn.execute('SELECT jti FROM token_blacklist')}
    assert remaining == {'jti_valid'}