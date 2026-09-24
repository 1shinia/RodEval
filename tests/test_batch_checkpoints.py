"""Durable resumable-batch checkpoint tests (SQLite only)."""

import pytest

from evalscope.service import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    db._db_path = None
    db._local.conn = None
    db.init_db(str(tmp_path))
    yield
    db._db_path = None
    db._local.conn = None


def _items():
    return [
        {'row_index': 0, 'model': 'same-model'},
        {'row_index': 1, 'model': 'same-model'},
    ]


def test_batch_checkpoint_tracks_duplicate_models_by_row_index():
    db.create_batch_job('batch_a', 'eval', 7, 'manifest-a', _items())

    db.update_batch_item('batch_a', 0, status='completed', task_id='eval_1')
    db.update_batch_item('batch_a', 1, status='interrupted', task_id='eval_2')
    job = db.get_batch_job('batch_a', user_id=7, batch_type='eval')

    assert job is not None
    assert [item['status'] for item in job['items']] == ['completed', 'interrupted']
    assert [item['row_index'] for item in job['items']] == [0, 1]


def test_cancelled_batch_can_be_claimed_for_resume_only_once():
    db.create_batch_job('batch_b', 'perf', 8, 'manifest-b', _items())
    db.update_batch_job('batch_b', status='cancelled', completed=1, errors=0)

    assert db.claim_batch_resume('batch_b', user_id=8, batch_type='perf', manifest_hash='manifest-b') == 'claimed'
    assert db.claim_batch_resume('batch_b', user_id=8, batch_type='perf', manifest_hash='manifest-b') == 'running'


def test_resume_rejects_wrong_owner_type_or_manifest():
    db.create_batch_job('batch_c', 'eval', 9, 'manifest-c', _items())
    db.update_batch_job('batch_c', status='cancelled')

    assert db.claim_batch_resume('batch_c', user_id=10, batch_type='eval', manifest_hash='manifest-c') == 'not_found'
    assert db.claim_batch_resume('batch_c', user_id=9, batch_type='perf', manifest_hash='manifest-c') == 'not_found'
    assert db.claim_batch_resume('batch_c', user_id=9, batch_type='eval', manifest_hash='other') == 'manifest_mismatch'


def test_checkpoint_never_persists_credentials():
    db.create_batch_job('batch_d', 'eval', 11, 'manifest-d', _items())
    job = db.get_batch_job('batch_d', user_id=11, batch_type='eval')

    serialized = repr(job).lower()
    assert 'api_key' not in serialized
    assert 'secret' not in serialized


def test_item_and_aggregate_checkpoint_commit_together():
    db.create_batch_job('batch_atomic', 'eval', 11, 'manifest-atomic', _items())

    db.checkpoint_batch_item(
        'batch_atomic',
        0,
        item_fields={'status': 'completed', 'task_id': 'eval_done'},
        job_fields={
            'completed': 1,
            'results_json': [{'task_id': 'eval_done', 'status': 'completed'}],
        },
    )

    job = db.get_batch_job('batch_atomic', user_id=11, batch_type='eval')
    assert job['completed'] == 1
    assert job['results'] == [{'task_id': 'eval_done', 'status': 'completed'}]
    assert job['items'][0]['status'] == 'completed'


def test_item_and_aggregate_checkpoint_roll_back_together_on_failure():
    db.create_batch_job('batch_rollback', 'perf', 11, 'manifest-rollback', _items())
    conn = db._get_conn()
    conn.execute('''
        CREATE TRIGGER fail_batch_job_update
        BEFORE UPDATE ON batch_jobs
        BEGIN
            SELECT RAISE(ABORT, 'forced aggregate failure');
        END
    ''')

    with pytest.raises(db.BatchCheckpointError, match='batch_rollback/0'):
        db.checkpoint_batch_item(
            'batch_rollback',
            0,
            item_fields={'status': 'completed', 'task_id': 'perf_done'},
            job_fields={'completed': 1},
        )

    conn.execute('DROP TRIGGER fail_batch_job_update')
    job = db.get_batch_job('batch_rollback', user_id=11, batch_type='perf')
    assert job['completed'] == 0
    assert job['items'][0]['status'] == 'pending'
    assert job['items'][0]['task_id'] == ''


def test_final_item_checkpoint_commits_terminal_job_state_atomically():
    db.create_batch_job('batch_terminal', 'eval', 11, 'manifest-terminal', [
        {'row_index': 0, 'model': 'only-model'},
    ])

    db.checkpoint_batch_item(
        'batch_terminal',
        0,
        item_fields={'status': 'completed', 'task_id': 'eval_done'},
        job_fields={
            'status': 'completed',
            'completed': 1,
            'results_json': [{'task_id': 'eval_done', 'status': 'completed'}],
        },
    )

    job = db.get_batch_job('batch_terminal', user_id=11, batch_type='eval')
    assert job['status'] == 'completed'
    assert job['completed'] == 1
    assert job['items'][0]['status'] == 'completed'


def test_recover_interrupted_batches_after_service_restart():
    db.create_batch_job('batch_e', 'eval', 12, 'manifest-e', [
        {'row_index': 0, 'model': 'done'},
        {'row_index': 1, 'model': 'active'},
        {'row_index': 2, 'model': 'waiting'},
    ])
    db.update_batch_item('batch_e', 0, status='completed')
    db.update_batch_item('batch_e', 1, status='running', task_id='eval_active')
    db.update_batch_job('batch_e', status='running', completed=1)

    assert db.recover_interrupted_batches() == 1
    job = db.get_batch_job('batch_e', user_id=12, batch_type='eval')

    assert job['status'] == 'stopped'
    assert [item['status'] for item in job['items']] == ['completed', 'interrupted', 'pending']
