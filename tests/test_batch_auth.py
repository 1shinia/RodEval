"""Batch endpoint auth isolation tests.

Covers the eval batch endpoints (upload / launch / status / stop) hardening:
batch_id path traversal rejection, cross-user launch/status/stop denial, owner
access, and the CSV upload size cap.

Constraint: like test_auth_isolation_e2e.py, importing evalscope.service pulls
in uvicorn/waitress through the perf module chain. Run this module under the
hermes env (``/root/anaconda3/envs/hermes/bin/python -m pytest``); the bare
evalscope env lacks those deps and will fail on import.
"""
import io
import os

import pytest

import evalscope.service.blueprints.eval as svc_eval
import evalscope.service.db as svc_db
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log


@pytest.fixture()
def clients(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils, svc_eval):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)
    # EVAL_BATCH_UPLOAD_DIR is computed at import time from OUTPUT_DIR; redirect
    # it too so uploads land in the temp dir instead of the real outputs/.
    monkeypatch.setattr(svc_eval, 'EVAL_BATCH_UPLOAD_DIR', os.path.join(root, '_eval_batch_uploads'))
    # Fresh in-memory batch state per test (the real one is a module global).
    monkeypatch.setattr(svc_eval, '_eval_batch_state', {})

    monkeypatch.setenv('EVALSCOPE_ADMIN_PASSWORD', 'testpass')
    from evalscope.service.app import create_app
    app = create_app(outputs=root)

    def _register(client, username):
        resp = client.post('/api/v1/auth/register', json={'username': username, 'password': 'pw123456'})
        assert resp.status_code == 201, resp.data
        body = resp.get_json()
        return body['token'], body['user']['id']

    client_a = app.test_client()
    client_b = app.test_client()
    token_a, uid_a = _register(client_a, 'batch_user_a')
    token_b, uid_b = _register(client_b, 'batch_user_b')
    client_a.environ_base['HTTP_AUTHORIZATION'] = 'Bearer ' + token_a
    client_b.environ_base['HTTP_AUTHORIZATION'] = 'Bearer ' + token_b
    return client_a, client_b, uid_a, uid_b


def _upload_csv(client) -> str:
    """Upload a small CSV as *client* and return the assigned batch_id."""
    resp = client.post(
        '/api/v1/eval/batch/upload',
        data={'file': (io.BytesIO(b'model,api,base_url,api_key\nm1,openai,http://x,sk\n'), 'models.csv')},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 200, resp.data
    return resp.get_json()['batch_id']


def _seed_batch_state(batch_id: str, user_id: int) -> None:
    """Insert a running-batch state as if it had been launched by *user_id*."""
    svc_eval._eval_batch_state[batch_id] = {
        'batch_id': batch_id,
        'user_id': user_id,
        'status': 'running',
        'total': 1,
        'completed': 0,
        'errors': 0,
        'current_model': '',
        'current_task_id': '',
        'results': [],
        'error_details': [],
        'cancel_requested': False,
    }


def test_launch_rejects_path_traversal(clients):
    client_a, *_ = clients
    resp = client_a.post('/api/v1/eval/batch/launch', json={'batch_id': '../../etc/passwd'})
    assert resp.status_code == 400


def test_other_user_cannot_launch_batch(clients):
    client_a, client_b, *_ = clients
    batch_id = _upload_csv(client_a)
    resp = client_b.post('/api/v1/eval/batch/launch', json={'batch_id': batch_id})
    assert resp.status_code == 404


def test_other_user_cannot_status_batch(clients):
    client_a, client_b, uid_a, _ = clients
    batch_id = _upload_csv(client_a)
    _seed_batch_state(batch_id, uid_a)
    assert client_b.get(f'/api/v1/eval/batch/status/{batch_id}').status_code == 404


def test_other_user_cannot_stop_batch(clients):
    client_a, client_b, uid_a, _ = clients
    batch_id = _upload_csv(client_a)
    _seed_batch_state(batch_id, uid_a)
    assert client_b.post(f'/api/v1/eval/batch/stop/{batch_id}').status_code == 404


def test_owner_can_status_own_batch(clients):
    client_a, _, uid_a, _ = clients
    batch_id = _upload_csv(client_a)
    _seed_batch_state(batch_id, uid_a)
    assert client_a.get(f'/api/v1/eval/batch/status/{batch_id}').status_code == 200


def test_upload_rejects_oversized_csv(clients):
    client_a, *_ = clients
    big = io.BytesIO(b'model,api\n' + b'x,openai\n' * 600000)  # ~6MB > 5MB cap
    resp = client_a.post(
        '/api/v1/eval/batch/upload',
        data={'file': (big, 'big.csv')},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 413


def test_upload_preview_hides_api_key_and_secures_temporary_csv(clients):
    client_a, *_ = clients
    secret = 'batch-secret-sentinel'
    resp = client_a.post(
        '/api/v1/eval/batch/upload',
        data={'file': (io.BytesIO(
            f'model,api,base_url,api_key\nm1,openai,http://x,{secret}\n'.encode()
        ), 'models.csv')},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 200, resp.data
    assert secret not in resp.get_data(as_text=True)
    batch_id = resp.get_json()['batch_id']
    saved = os.path.join(svc_eval.EVAL_BATCH_UPLOAD_DIR, f'{batch_id}.csv')
    assert os.path.isfile(saved)
    assert oct(os.stat(saved).st_mode & 0o777) == '0o600'


class _DormantThread:
    def __init__(self, *, target, daemon):
        self.target = target

    def start(self):
        pass


def test_launch_persists_checkpoint_and_removes_credential_file(clients, monkeypatch):
    client_a, _, uid_a, _ = clients
    batch_id = _upload_csv(client_a)
    monkeypatch.setattr(svc_eval.threading, 'Thread', _DormantThread)

    resp = client_a.post('/api/v1/eval/batch/launch', json={
        'batch_id': batch_id,
        'eval_backend': 'Native',
        'datasets': ['gsm8k'],
    })

    assert resp.status_code == 200, resp.data
    assert not os.path.exists(os.path.join(svc_eval.EVAL_BATCH_UPLOAD_DIR, f'{batch_id}.csv'))
    job = svc_db.get_batch_job(batch_id, user_id=uid_a, batch_type='eval')
    assert job is not None
    assert job['status'] == 'running'
    assert [item['status'] for item in job['items']] == ['pending']


def test_status_uses_durable_cancelled_checkpoint(clients):
    client_a, _, uid_a, _ = clients
    svc_db.create_batch_job('durable_eval', 'eval', uid_a, 'manifest', [{'row_index': 0, 'model': 'm1'}])
    svc_db.update_batch_job('durable_eval', status='cancelled')

    resp = client_a.get('/api/v1/eval/batch/status/durable_eval')

    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'cancelled'
    assert resp.get_json()['resumable'] is True


def test_resume_reuses_original_batch_id_and_rejects_second_claim(clients, monkeypatch):
    client_a, _, _, _ = clients
    original = _upload_csv(client_a)
    monkeypatch.setattr(svc_eval.threading, 'Thread', _DormantThread)
    launch = client_a.post('/api/v1/eval/batch/launch', json={
        'batch_id': original, 'eval_backend': 'Native', 'datasets': ['gsm8k'],
    })
    assert launch.status_code == 200
    svc_db.update_batch_job(original, status='cancelled')
    svc_eval._eval_batch_state.clear()

    upload_id = _upload_csv(client_a)
    payload = {
        'batch_id': original,
        'upload_id': upload_id,
        'eval_backend': 'Native',
        'datasets': ['gsm8k'],
    }
    resumed = client_a.post('/api/v1/eval/batch/resume', json=payload)
    duplicate = client_a.post('/api/v1/eval/batch/resume', json=payload)

    assert resumed.status_code == 200, resumed.data
    assert resumed.get_json()['batch_id'] == original
    assert duplicate.status_code == 409
