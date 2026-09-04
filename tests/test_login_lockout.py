"""Login brute-force protection: per-(IP, username) lockout after repeated failures."""
import pytest

import evalscope.service.blueprints.auth as svc_auth
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)
    monkeypatch.setenv('EVALSCOPE_ADMIN_PASSWORD', 'adminpass')

    # Tight thresholds so tests do not need 5 real attempts.
    monkeypatch.setattr(svc_auth, '_LOGIN_MAX_ATTEMPTS', 3)
    monkeypatch.setattr(svc_auth, '_LOGIN_WINDOW_SECONDS', 300)
    monkeypatch.setattr(svc_auth, '_LOGIN_LOCKOUT_SECONDS', 300)
    # Fresh guard state per test (module-level dicts are shared).
    svc_auth._login_failures.clear()
    svc_auth._login_locks.clear()

    from evalscope.service.app import create_app
    app = create_app(outputs=root)
    svc_auth.ensure_admin_user()
    c = app.test_client()
    resp = c.post('/api/v1/auth/register', json={'username': 'alice', 'password': 'secret123'})
    assert resp.status_code == 201, resp.data
    return c


def _login(c, username='alice', password='wrongpass'):
    return c.post('/api/v1/auth/login', json={'username': username, 'password': password})


def test_failures_below_threshold_still_allowed(client):
    for _ in range(2):
        assert _login(client).status_code == 401
    # Budget not exhausted: correct password still works.
    assert _login(client, password='secret123').status_code == 200


def test_lockout_after_max_attempts(client):
    for _ in range(3):
        assert _login(client).status_code == 401
    resp = _login(client)
    assert resp.status_code == 429
    assert int(resp.headers['Retry-After']) > 0
    assert '尝试次数过多' in resp.get_json()['error']


def test_locked_even_with_correct_password(client):
    for _ in range(3):
        assert _login(client).status_code == 401
    assert _login(client, password='secret123').status_code == 429


def test_unknown_username_also_counts_and_locks(client):
    for _ in range(3):
        assert _login(client, username='ghost').status_code == 401
    assert _login(client, username='ghost').status_code == 429


def test_lockout_is_per_username(client):
    for _ in range(3):
        assert _login(client).status_code == 401
    assert _login(client).status_code == 429
    # A different username from the same IP is unaffected.
    assert _login(client, username='admin', password='adminpass').status_code == 200


def test_successful_login_clears_failure_budget(client):
    for _ in range(2):
        assert _login(client).status_code == 401
    assert _login(client, password='secret123').status_code == 200
    # Budget was reset: two more failures are allowed again.
    for _ in range(2):
        assert _login(client).status_code == 401
    assert _login(client, password='secret123').status_code == 200


def test_lock_expires(client, monkeypatch):
    for _ in range(3):
        assert _login(client).status_code == 401
    assert _login(client).status_code == 429
    # Fast-forward past the lockout window.
    key = next(iter(svc_auth._login_locks))
    svc_auth._login_locks[key] -= 301
    assert _login(client, password='secret123').status_code == 200
