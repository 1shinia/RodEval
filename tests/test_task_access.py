"""Task-artifact ownership isolation tests (user A must not read user B's artifacts).

Covers the shared task_access policy used by report/SSE/file routes:
DB metadata row is authoritative, ``.owner`` marker covers the startup race,
legacy dirs without either are admin-only, and task_id is re-validated.
"""
import os

import pytest

from evalscope.service import db
from evalscope.service.task_access import task_artifact_owned_by


@pytest.fixture()
def meta_db(tmp_path):
    db.init_db(str(tmp_path))
    conn = db._get_conn()
    # users: id=1 admin (seeded), id=2 user A, id=3 user B
    conn.execute(
        "INSERT INTO users(username, password_hash, role, created_at) VALUES ('alice', 'x', 'user', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO users(username, password_hash, role, created_at) VALUES ('bob', 'x', 'user', '2026-01-01')"
    )
    conn.commit()
    return tmp_path, db


def _seed_eval_report(meta_db, task_id: str, user_id: int):
    _, db = meta_db
    conn = db._get_conn()
    conn.execute(
        'INSERT INTO eval_reports(task_id, model_name, dataset_name, user_id) VALUES (?, ?, ?, ?)',
        (task_id, 'model-x', 'dataset-y', user_id),
    )
    conn.commit()


def test_owner_can_read_own_task(meta_db):
    _seed_eval_report(meta_db, 'eval_aaa', 2)
    assert task_artifact_owned_by('eval_aaa', ('eval_reports',), user_id=2, is_admin=False, output_dir='') is True


def test_other_user_cannot_read_task(meta_db):
    _seed_eval_report(meta_db, 'eval_aaa', 2)
    assert task_artifact_owned_by('eval_aaa', ('eval_reports',), user_id=3, is_admin=False, output_dir='') is False


def test_admin_does_not_bypass_db_ownership(meta_db):
    # DB row exists and belongs to alice: even admin is denied (consistent with
    # the historical check_task_ownership semantics).
    _seed_eval_report(meta_db, 'eval_aaa', 2)
    assert task_artifact_owned_by('eval_aaa', ('eval_reports',), user_id=1, is_admin=True, output_dir='') is False


def test_legacy_dir_without_metadata_is_admin_only(meta_db, tmp_path):
    task_dir = tmp_path / 'eval_legacy'
    task_dir.mkdir()
    assert task_artifact_owned_by('eval_legacy', ('eval_reports', 'task_state'), user_id=2, is_admin=False, output_dir=str(tmp_path)) is False
    assert task_artifact_owned_by('eval_legacy', ('eval_reports', 'task_state'), user_id=1, is_admin=True, output_dir=str(tmp_path)) is True


def test_owner_marker_fallback_before_db_index(meta_db, tmp_path):
    # Simulate a task that started but has not been indexed in the DB yet.
    task_dir = tmp_path / 'eval_starting'
    task_dir.mkdir()
    with open(task_dir / '.owner', 'w') as f:
        f.write('2')
    assert task_artifact_owned_by('eval_starting', ('eval_reports', 'task_state'), user_id=2, is_admin=False, output_dir=str(tmp_path)) is True
    assert task_artifact_owned_by('eval_starting', ('eval_reports', 'task_state'), user_id=3, is_admin=False, output_dir=str(tmp_path)) is False


def test_invalid_task_id_rejected(meta_db, tmp_path):
    for bad in ('../etc/passwd', 'a/b', '..', ''):
        assert task_artifact_owned_by(bad, ('eval_reports',), user_id=2, is_admin=False, output_dir=str(tmp_path)) is False


def test_unsupported_table_raises(meta_db):
    with pytest.raises(ValueError):
        task_artifact_owned_by('eval_aaa', ('not_a_table',), user_id=2, is_admin=False, output_dir='')


def test_task_state_table_also_checked(meta_db):
    _, db = meta_db
    conn = db._get_conn()
    conn.execute(
        "INSERT INTO task_state(task_id, task_type, status, started_at, updated_at, user_id) "
        "VALUES ('eval_running', 'eval', 'running', '2026-01-01', '2026-01-01', 3)"
    )
    conn.commit()
    assert task_artifact_owned_by('eval_running', ('eval_reports', 'task_state'), user_id=3, is_admin=False, output_dir='') is True
    assert task_artifact_owned_by('eval_running', ('eval_reports', 'task_state'), user_id=2, is_admin=False, output_dir='') is False


def test_task_registry_authorizes_unfinished_owner(meta_db):
    """Interrupted tasks may have registry ownership before any result row."""
    _, db = meta_db
    conn = db._get_conn()
    conn.execute(
        "INSERT INTO task_registry(task_id, task_kind, user_id, created_at) "
        "VALUES ('eval_unfinished', 'eval', 2, '2026-01-01')"
    )
    conn.commit()
    assert task_artifact_owned_by(
        'eval_unfinished', ('task_registry', 'task_state', 'eval_reports'),
        user_id=2, is_admin=False, output_dir=''
    ) is True
    assert task_artifact_owned_by(
        'eval_unfinished', ('task_registry', 'task_state', 'eval_reports'),
        user_id=3, is_admin=False, output_dir=''
    ) is False
