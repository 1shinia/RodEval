# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for SQLite lock hardening in evalscope.service.db.

Covers: upsert_eval_report basic write, clean failure + connection reuse
under a held write lock, and checkpoint_db tolerance.  Uses an isolated
temp DB so it never touches the real metadata store.
"""
import os
import sqlite3
import tempfile

import pytest

from evalscope.service import db as _db


@pytest.fixture()
def iso_db():
    """Point the db module at a fresh temp DB per test, then reset.

    Teardown must NOT re-point at the real outputs DB: init_db() there would
    run pending migrations against production during tests (a latent hazard
    that surfaced with migration v9 — test teardown applied v9 to the live
    evalscope_meta.db). Resetting to None keeps the module uninitialized.
    """
    tmp = tempfile.mkdtemp()
    _db._local.conn = None
    _db.init_db(tmp)
    yield tmp
    _db._db_path = None
    _db._local.conn = None


def test_upsert_eval_report_basic(iso_db):
    _db.upsert_eval_report(
        task_id='t1', model_name='m', dataset_name='d', score=0.5,
        num_samples=1, timestamp='2026-08-21T00:00:00', dataset_scores={'d': 0.5},
        eval_backend='Native', user_id=1,
    )
    row = _db._get_conn().execute(
        "SELECT task_id, model_name, dataset_name, score, eval_backend FROM eval_reports WHERE task_id='t1'"
    ).fetchone()
    assert tuple(row) == ('t1', 'm', 'd', 0.5, 'Native')


def test_upsert_fails_cleanly_on_lock_then_recovers(iso_db):
    """A held write lock makes upsert raise (fast, 1s busy timeout) and the
    module connection stays usable: after the lock is released the same
    connection writes successfully."""
    db_path = os.path.join(iso_db, 'evalscope_meta.db')
    _db._local.conn = sqlite3.connect(db_path, timeout=1)
    _db._local.conn.row_factory = sqlite3.Row

    holder = sqlite3.connect(db_path, timeout=30)
    holder.execute('BEGIN IMMEDIATE')  # long-held write lock
    try:
        with pytest.raises(sqlite3.OperationalError):
            _db.upsert_eval_report(
                task_id='t2', model_name='m', dataset_name='d', score=0.8,
                num_samples=5, timestamp='x', eval_backend='Native', user_id=1,
            )
    finally:
        holder.rollback()
        holder.close()

    # Same module connection must now succeed (rollback left it clean).
    _db.upsert_eval_report(
        task_id='t2', model_name='m', dataset_name='d', score=0.8,
        num_samples=5, timestamp='x', eval_backend='Native', user_id=1,
    )
    row = _db._get_conn().execute("SELECT task_id, score FROM eval_reports WHERE task_id='t2'").fetchone()
    assert tuple(row) == ('t2', 0.8)


def test_checkpoint_db_normal_and_tolerant(iso_db):
    r = _db.checkpoint_db()
    assert isinstance(r, dict) and {'busy', 'log', 'checkpointed'} <= set(r)
    # No DB initialized -> must not raise, returns busy=-1 sentinel.
    _db._db_path = None
    _db._local.conn = None
    r2 = _db.checkpoint_db()
    assert r2['busy'] == -1
