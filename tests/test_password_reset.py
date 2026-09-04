"""Password reset token flow: admin generates a one-time token, user resets."""
import pytest

import evalscope.service.blueprints.auth as svc_auth
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log


@pytest.fixture()
def clients(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)
    monkeypatch.setenv('EVALSCOPE_ADMIN_PASSWORD', 'adminpass')

    from evalscope.service.app import create_app
    app = create_app(outputs=root)
    svc_auth.ensure_admin_user()

    anon = app.test_client()
    resp = anon.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    assert resp.status_code == 200, resp.data
    admin_token = resp.get_json()['token']

    resp = anon.post('/api/v1/auth/register', json={'username': 'alice', 'password': 'oldpass123'})
    assert resp.status_code == 201, resp.data
    alice_id = resp.get_json()['user']['id']

    admin_client = app.test_client()
    admin_client.environ_base['HTTP_AUTHORIZATION'] = f'Bearer {admin_token}'

    alice_client = app.test_client()
    alice_client.environ_base['HTTP_AUTHORIZATION'] = (
        f'Bearer {anon.post("/api/v1/auth/login", json={"username": "alice", "password": "oldpass123"}).get_json()["token"]}'
    )

    return app, admin_client, alice_client, alice_id


def test_non_admin_cannot_create_reset_token(clients):
    _, _, alice_client, alice_id = clients
    resp = alice_client.post(f'/api/v1/auth/users/{alice_id}/reset-token')
    assert resp.status_code == 403


def test_admin_can_create_reset_token(clients):
    app, admin_client, _, alice_id = clients
    resp = admin_client.post(f'/api/v1/auth/users/{alice_id}/reset-token')
    assert resp.status_code == 201
    body = resp.get_json()
    assert body['token'] and body['expires_at']


def test_reset_password_flow(clients):
    app, admin_client, _, alice_id = clients
    token = admin_client.post(f'/api/v1/auth/users/{alice_id}/reset-token').get_json()['token']

    # old password still works before reset
    assert app.test_client().post('/api/v1/auth/login', json={'username': 'alice', 'password': 'oldpass123'}).status_code == 200

    resp = app.test_client().post('/api/v1/auth/reset-password', json={'token': token, 'password': 'newpass456'})
    assert resp.status_code == 200, resp.data

    # new password works, old one is gone
    assert app.test_client().post('/api/v1/auth/login', json={'username': 'alice', 'password': 'newpass456'}).status_code == 200
    assert app.test_client().post('/api/v1/auth/login', json={'username': 'alice', 'password': 'oldpass123'}).status_code == 401


def test_reset_token_single_use(clients):
    app, admin_client, _, alice_id = clients
    token = admin_client.post(f'/api/v1/auth/users/{alice_id}/reset-token').get_json()['token']

    first = app.test_client().post('/api/v1/auth/reset-password', json={'token': token, 'password': 'newpass456'})
    assert first.status_code == 200
    second = app.test_client().post('/api/v1/auth/reset-password', json={'token': token, 'password': 'another789'})
    assert second.status_code == 400


def test_reset_invalid_token_rejected(clients):
    app, _, _, _ = clients
    resp = app.test_client().post('/api/v1/auth/reset-password', json={'token': 'bogus', 'password': 'newpass456'})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == '链接无效或已过期'
