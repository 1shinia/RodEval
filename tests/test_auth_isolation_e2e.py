"""End-to-end auth isolation: user A must not read user B's eval artifacts.

Covers the new cookie-auth model (login sets HttpOnly cookie; GET/HEAD may use
it) and the shared task ownership policy on report / SSE routes.  Unauthenticated
requests to task artifacts must 401; cross-user access must 404.
"""
import pytest

import evalscope.service.blueprints.eval as svc_eval
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log

TASK_A = 'eval_e2e_a'
TASK_B = 'eval_e2e_b'


def _seed_task(root, task_id: str, user_id: int):
    task_dir = root / task_id
    reports = task_dir / 'reports'
    logs = task_dir / 'logs'
    reports.mkdir(parents=True)
    logs.mkdir(parents=True)
    (reports / 'report.html').write_text(f'<html>report {task_id}</html>')
    (logs / 'eval_log.log').write_text('line1\nline2\n')


@pytest.fixture()
def clients(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils, svc_eval):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)

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
    token_a, uid_a = _register(client_a, 'user_a')
    token_b, uid_b = _register(client_b, 'user_b')

    # Seed two tasks owned by different users (DB metadata row + artifact dirs).
    from evalscope.service import db
    conn = db._get_conn()
    for task_id, uid in ((TASK_A, uid_a), (TASK_B, uid_b)):
        conn.execute(
            'INSERT INTO eval_reports(task_id, model_name, dataset_name, user_id) VALUES (?, ?, ?, ?)',
            (task_id, 'model-x', 'dataset-y', uid),
        )
        _seed_task(tmp_path, task_id, uid)
    conn.commit()

    client_a.environ_base['HTTP_AUTHORIZATION'] = f'Bearer {token_a}'
    client_b.environ_base['HTTP_AUTHORIZATION'] = f'Bearer {token_b}'
    return client_a, client_b, app


def test_unauthenticated_artifact_requests_rejected(clients):
    client_a, _, app = clients
    anon = app.test_client()
    assert anon.get(f'/api/v1/eval/report?task_id={TASK_A}').status_code == 401
    assert anon.get(f'/api/v1/eval/log?task_id={TASK_A}').status_code == 401
    assert anon.get(f'/api/v1/eval/log/stream?task_id={TASK_A}').status_code == 401


def test_owner_can_read_own_report(clients):
    client_a, _, _ = clients
    resp = client_a.get(f'/api/v1/eval/report?task_id={TASK_A}')
    assert resp.status_code == 200
    assert 'report eval_e2e_a' in resp.data.decode()


def test_other_user_cannot_read_report(clients):
    _, client_b, _ = clients
    resp = client_b.get(f'/api/v1/eval/report?task_id={TASK_A}')
    assert resp.status_code == 404
    assert b'Task not found' in resp.data


def test_other_user_cannot_read_log_or_sse(clients):
    _, client_b, _ = clients
    assert client_b.get(f'/api/v1/eval/log?task_id={TASK_A}').status_code == 404
    assert client_b.get(f'/api/v1/eval/log/stream?task_id={TASK_A}').status_code == 404


def test_cookie_auth_works_for_get(clients):
    client_a, _, app = clients
    cookie_client = app.test_client()
    # login sets the HttpOnly cookie; no Authorization header on the GET.
    resp = cookie_client.post('/api/v1/auth/login', json={'username': 'user_a', 'password': 'pw123456'})
    assert resp.status_code == 200
    assert resp.headers.get('Set-Cookie', '').startswith('evalscope_auth=')
    assert 'HttpOnly' in resp.headers.get('Set-Cookie', '')
    assert 'SameSite=Lax' in resp.headers.get('Set-Cookie', '')

    report = cookie_client.get(f'/api/v1/eval/report?task_id={TASK_A}')
    assert report.status_code == 200
    assert 'report eval_e2e_a' in report.data.decode()
    # Cookie does not grant access to user B's task.
    assert cookie_client.get(f'/api/v1/eval/report?task_id={TASK_B}').status_code == 404


def test_path_traversal_rejected(clients):
    client_a, _, _ = clients
    assert client_a.get('/api/v1/eval/report?task_id=../etc/passwd').status_code == 404
    assert client_a.get('/api/v1/eval/log/stream?task_id=..%2F..%2Fetc%2Fpasswd').status_code == 404
