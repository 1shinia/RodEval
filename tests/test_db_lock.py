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


def test_new_task_id_available_checks_db_and_filesystem(iso_db):
    assert _db.new_task_id_available(iso_db, 'fresh-task') is True

    _db.upsert_perf_task(
        task_id='existing-db', model='m', api='openai', dataset='d', runs=1,
        has_report=False, timestamp='2026-08-26', user_id=1,
    )
    assert _db.new_task_id_available(iso_db, 'existing-db') is False

    os.makedirs(os.path.join(iso_db, 'existing-dir'))
    assert _db.new_task_id_available(iso_db, 'existing-dir') is False


def test_task_id_reservation_is_atomic_and_releasable(iso_db):
    assert _db.reserve_task_id('reserved-task', 'eval', 7) is True
    assert _db.reserve_task_id('reserved-task', 'eval', 7) is False
    row = _db._get_conn().execute(
        "SELECT task_kind, user_id FROM task_registry WHERE task_id='reserved-task'"
    ).fetchone()
    assert tuple(row) == ('eval', 7)

    # A reservation that never produced metadata/artifacts can be returned.
    assert _db.release_task_id_reservation('reserved-task', user_id=7) is True
    assert _db.new_task_id_available(iso_db, 'reserved-task') is True


def test_task_registry_rejects_cross_kind_reuse(iso_db):
    _db.upsert_eval_report(
        task_id='global-id', model_name='m', dataset_name='d', score=0.1,
        num_samples=1, timestamp='2026-08-26', eval_backend='Native', user_id=7,
    )
    with pytest.raises(ValueError, match='registered as eval'):
        _db.upsert_perf_task(
            task_id='global-id', model='m', api='a', dataset='d', runs=1,
            has_report=False, timestamp='2026-08-26', user_id=7,
        )


def test_eval_dataset_relation_uses_exact_membership_and_scores(iso_db):
    _db.upsert_eval_report(
        task_id='multi', model_name='m', dataset_name='mmlu_pro, gsm8k', score=0.75,
        num_samples=10, timestamp='2026-08-26',
        dataset_scores={'mmlu_pro': 0.8, 'gsm8k': 0.7},
        eval_backend='Native', user_id=1,
    )
    _db.upsert_eval_report(
        task_id='exact', model_name='m', dataset_name='mmlu', score=0.9,
        num_samples=10, timestamp='2026-08-27',
        dataset_scores={'mmlu': 0.9}, eval_backend='Native', user_id=1,
    )

    items, total, _models, datasets = _db.query_eval_reports(datasets='mmlu', user_id=1)
    assert total == 1
    assert len(items) == 1 and items[0]['name'].startswith('exact@@')
    assert datasets == ['gsm8k', 'mmlu', 'mmlu_pro']

    rows = _db._get_conn().execute(
        '''SELECT dataset_name, score, position
           FROM eval_report_datasets WHERE task_id = 'multi'
           ORDER BY position'''
    ).fetchall()
    assert [tuple(row) for row in rows] == [('mmlu_pro', 0.8, 0), ('gsm8k', 0.7, 1)]


def test_task_owner_cannot_be_reassigned_by_upsert(iso_db):
    _db.upsert_eval_report(
        task_id='owned', model_name='m', dataset_name='d', score=0.1,
        num_samples=1, timestamp='2026-08-26', eval_backend='Native', user_id=7,
    )
    with pytest.raises(PermissionError):
        _db.upsert_eval_report(
            task_id='owned', model_name='other', dataset_name='d2', score=0.9,
            num_samples=1, timestamp='2026-08-26', eval_backend='Native', user_id=8,
        )
    row = _db._get_conn().execute(
        "SELECT user_id, model_name FROM eval_reports WHERE task_id='owned'"
    ).fetchone()
    assert tuple(row) == (7, 'm')


def test_owner_marker_refuses_cross_user_replacement(iso_db):
    task_dir = os.path.join(iso_db, 'owner-marker')
    _db.write_owner_marker(task_dir, 7)
    _db.write_owner_marker(task_dir, 7)  # same-owner resume is fine
    with pytest.raises(PermissionError):
        _db.write_owner_marker(task_dir, 8)
    assert _db.read_owner(task_dir) == 7


def test_init_db_switches_thread_connection_to_new_path(tmp_path):
    """Re-initialising for another output dir must not keep using the old DB."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    try:
        _db.init_db(str(first))
        conn1 = _db._get_conn()
        _db.upsert_perf_task(
            task_id='only-first', model='m', api='a', dataset='d', runs=1,
            has_report=False, timestamp='2026-08-26', user_id=1,
        )

        _db.init_db(str(second))
        conn2 = _db._get_conn()
        assert conn2 is not conn1
        assert os.path.isfile(second / 'evalscope_meta.db')
        assert conn2.execute(
            "SELECT COUNT(*) FROM perf_tasks WHERE task_id='only-first'"
        ).fetchone()[0] == 0
    finally:
        conn = getattr(_db._local, 'conn', None)
        if conn is not None:
            conn.close()
        _db._local.conn = None
        _db._db_path = None


def test_perf_task_owner_cannot_be_reassigned_by_upsert(iso_db):
    _db.upsert_perf_task(
        task_id='owned-perf', model='m', api='a', dataset='d', runs=1,
        has_report=False, timestamp='2026-08-26', user_id=7,
    )
    with pytest.raises(PermissionError):
        _db.upsert_perf_task(
            task_id='owned-perf', model='other', api='b', dataset='d2', runs=2,
            has_report=True, timestamp='2026-08-27', user_id=8,
        )
    row = _db._get_conn().execute(
        "SELECT user_id, model FROM perf_tasks WHERE task_id='owned-perf'"
    ).fetchone()
    assert tuple(row) == (7, 'm')
