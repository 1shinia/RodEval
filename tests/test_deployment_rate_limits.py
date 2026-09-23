"""Deployment registration policy and request-rate limiting."""

import datetime

import pytest

import evalscope.service.blueprints.auth as svc_auth
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log


@pytest.fixture()
def app_factory(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)
    monkeypatch.setenv('EVALSCOPE_ADMIN_PASSWORD', 'adminpass')

    def _make(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        from evalscope.service.app import create_app
        app = create_app(outputs=root)
        app.config['TESTING'] = True
        return app

    return _make


def test_registration_is_admin_only_by_default(app_factory, monkeypatch):
    monkeypatch.delenv('REGISTRATION_MODE', raising=False)
    app = app_factory()

    response = app.test_client().post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123'},
    )

    assert response.status_code == 403
    assert response.get_json()['error'] == '公开注册已关闭，请联系管理员创建账号'


def test_public_registration_can_be_explicitly_enabled(app_factory):
    app = app_factory(REGISTRATION_MODE='public')

    response = app.test_client().post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123'},
    )

    assert response.status_code == 201


def test_invite_registration_consumes_valid_code_once(app_factory):
    app = app_factory(REGISTRATION_MODE='invite')
    client = app.test_client()

    admin_login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    admin_headers = {'Authorization': f"Bearer {admin_login.get_json()['token']}"}
    generated = client.post(
        '/api/v1/auth/invites',
        json={'max_uses': 1},
        headers=admin_headers,
    )
    assert generated.status_code == 201
    code = generated.get_json()['code']

    first = client.post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123', 'invite_code': code},
    )
    second = client.post(
        '/api/v1/auth/register',
        json={'username': 'bob', 'password': 'secret123', 'invite_code': code},
    )

    assert first.status_code == 201
    assert second.status_code == 400
    assert second.get_json()['error'] == '邀请码无效或已过期'


def test_invite_registration_rejects_missing_or_expired_code(app_factory):
    app = app_factory(REGISTRATION_MODE='invite')
    client = app.test_client()
    response = client.post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123'},
    )
    assert response.status_code == 400
    assert response.get_json()['error'] == '邀请码无效或已过期'


def test_invite_code_is_hashed_at_rest(app_factory):
    app = app_factory(REGISTRATION_MODE='invite')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}
    code = client.post('/api/v1/auth/invites', json={}, headers=headers).get_json()['code']

    conn = svc_auth._get_conn()
    row = conn.execute('SELECT code_hash FROM registration_invites').fetchone()
    assert row['code_hash'] != code
    assert code not in row['code_hash']


def test_invite_defaults_to_24_hour_expiry(app_factory):
    app = app_factory(REGISTRATION_MODE='invite')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}

    response = client.post('/api/v1/auth/invites', json={}, headers=headers)

    assert response.status_code == 201
    expires_at = datetime.datetime.fromisoformat(response.get_json()['expires_at'])
    remaining = expires_at - datetime.datetime.now(datetime.timezone.utc)
    assert datetime.timedelta(hours=23, minutes=59) < remaining <= datetime.timedelta(hours=24)


def test_expired_invite_is_rejected(app_factory):
    app = app_factory(REGISTRATION_MODE='invite')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}
    code = client.post('/api/v1/auth/invites', json={}, headers=headers).get_json()['code']
    svc_auth._write(lambda conn: conn.execute(
        "UPDATE registration_invites SET expires_at = '2000-01-01T00:00:00+00:00'"
    ))

    response = client.post('/api/v1/auth/register', json={
        'username': 'alice', 'password': 'secret123', 'invite_code': code,
    })
    assert response.status_code == 400
    assert response.get_json()['error'] == '邀请码无效或已过期'


def test_invite_registration_is_rate_limited_per_ip(app_factory):
    app = app_factory(
        REGISTRATION_MODE='invite',
        RATE_LIMIT_REGISTER_REQUESTS='1',
        RATE_LIMIT_REGISTER_WINDOW_SECONDS='3600',
    )
    client = app.test_client()
    first = client.post('/api/v1/auth/register', json={
        'username': 'alice', 'password': 'secret123', 'invite_code': 'invalid',
    })
    second = client.post('/api/v1/auth/register', json={
        'username': 'bob', 'password': 'secret123', 'invite_code': 'invalid',
    })
    assert first.status_code == 400
    assert second.status_code == 429
    assert second.headers['Retry-After']


def test_invite_generation_is_admin_only(app_factory):
    app = app_factory(REGISTRATION_MODE='invite')
    response = app.test_client().post('/api/v1/auth/invites', json={'max_uses': 1})
    assert response.status_code == 401


def test_invalid_rate_limit_config_fails_at_startup(app_factory):
    with pytest.raises(RuntimeError, match='RATE_LIMIT_EVAL_REQUESTS'):
        app_factory(RATE_LIMIT_EVAL_REQUESTS='not-an-integer')


def test_config_exposes_registration_mode_without_internal_rate_limits(app_factory):
    app = app_factory(REGISTRATION_MODE='public', RATE_LIMIT_REGISTER_REQUESTS='7')

    body = app.test_client().get('/api/v1/config').get_json()

    assert body['registration_mode'] == 'public'
    assert 'rate_limits' not in body


def test_admin_can_change_registration_mode_without_restart(app_factory):
    app = app_factory(REGISTRATION_MODE='admin_only', REGISTRATION_MODE_LOCKED='false')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}

    changed = client.put(
        '/api/v1/auth/settings/registration',
        json={'mode': 'public'},
        headers=headers,
    )

    assert changed.status_code == 200
    assert changed.get_json() == {'registration_mode': 'public', 'registration_mode_locked': False}
    assert client.get('/api/v1/config').get_json()['registration_mode'] == 'public'
    registered = client.post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123'},
    )
    assert registered.status_code == 201
    row = svc_auth._get_conn().execute(
        "SELECT value, updated_by FROM system_settings WHERE key = 'registration_mode'"
    ).fetchone()
    assert (row['value'], row['updated_by']) == ('public', 1)


def test_registration_mode_update_requires_admin(app_factory):
    app = app_factory(REGISTRATION_MODE_LOCKED='false')
    response = app.test_client().put(
        '/api/v1/auth/settings/registration', json={'mode': 'public'}
    )
    assert response.status_code == 401
    assert app.test_client().get('/api/v1/config').get_json()['registration_mode'] == 'admin_only'


def test_regular_user_cannot_change_registration_mode(app_factory):
    app = app_factory(REGISTRATION_MODE_LOCKED='false')
    client = app.test_client()
    admin_login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    admin_headers = {'Authorization': f"Bearer {admin_login.get_json()['token']}"}
    assert client.post(
        '/api/v1/auth/users',
        json={'username': 'alice', 'password': 'secret123', 'role': 'user'},
        headers=admin_headers,
    ).status_code == 201
    user_login = client.post('/api/v1/auth/login', json={'username': 'alice', 'password': 'secret123'})
    user_headers = {'Authorization': f"Bearer {user_login.get_json()['token']}"}

    response = client.put(
        '/api/v1/auth/settings/registration', json={'mode': 'public'}, headers=user_headers
    )

    assert response.status_code == 403
    assert client.get('/api/v1/config').get_json()['registration_mode'] == 'admin_only'


def test_registration_mode_lock_forces_deployment_value(app_factory):
    app = app_factory(REGISTRATION_MODE='invite', REGISTRATION_MODE_LOCKED='true')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}

    changed = client.put(
        '/api/v1/auth/settings/registration', json={'mode': 'public'}, headers=headers
    )

    assert changed.status_code == 403
    assert '服务器配置锁定' in changed.get_json()['error']
    config = client.get('/api/v1/config').get_json()
    assert config['registration_mode'] == 'invite'
    assert config['registration_mode_locked'] is True


def test_registration_mode_rejects_invalid_value(app_factory):
    app = app_factory(REGISTRATION_MODE_LOCKED='false')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}

    response = client.put(
        '/api/v1/auth/settings/registration', json={'mode': 'anything'}, headers=headers
    )

    assert response.status_code == 400
    assert client.get('/api/v1/config').get_json()['registration_mode'] == 'admin_only'


def test_registration_mode_survives_app_restart(app_factory):
    app = app_factory(REGISTRATION_MODE='admin_only', REGISTRATION_MODE_LOCKED='false')
    client = app.test_client()
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'adminpass'})
    headers = {'Authorization': f"Bearer {login.get_json()['token']}"}
    assert client.put(
        '/api/v1/auth/settings/registration', json={'mode': 'public'}, headers=headers
    ).status_code == 200

    restarted = app_factory(REGISTRATION_MODE='admin_only', REGISTRATION_MODE_LOCKED='false')

    assert restarted.test_client().get('/api/v1/config').get_json()['registration_mode'] == 'public'


def test_corrupt_registration_mode_fails_closed(app_factory):
    app = app_factory(REGISTRATION_MODE='admin_only', REGISTRATION_MODE_LOCKED='false')
    svc_auth._write(lambda conn: conn.execute(
        "UPDATE system_settings SET value = 'corrupt' WHERE key = 'registration_mode'"
    ))

    config = app.test_client().get('/api/v1/config').get_json()
    response = app.test_client().post(
        '/api/v1/auth/register', json={'username': 'alice', 'password': 'secret123'}
    )

    assert config['registration_mode'] == 'admin_only'
    assert response.status_code == 403


def test_corrupt_registration_mode_does_not_fallback_to_public(app_factory):
    app = app_factory(REGISTRATION_MODE='public', REGISTRATION_MODE_LOCKED='false')
    svc_auth._write(lambda conn: conn.execute(
        "UPDATE system_settings SET value = 'corrupt' WHERE key = 'registration_mode'"
    ))

    config = app.test_client().get('/api/v1/config').get_json()
    response = app.test_client().post(
        '/api/v1/auth/register', json={'username': 'alice', 'password': 'secret123'}
    )

    assert config['registration_mode'] == 'admin_only'
    assert response.status_code == 403


def test_invalid_registration_lock_config_fails_at_startup(app_factory):
    with pytest.raises(RuntimeError, match='REGISTRATION_MODE_LOCKED must be a boolean'):
        app_factory(REGISTRATION_MODE_LOCKED='sometimes')


def test_untrusted_peer_cannot_spoof_client_ip(monkeypatch):
    from flask import Flask
    from evalscope.service.rate_limit import get_client_ip

    monkeypatch.setenv('TRUSTED_PROXIES', '127.0.0.1,::1')
    app = Flask(__name__)
    with app.test_request_context(
        '/', environ_base={'REMOTE_ADDR': '203.0.113.9'},
        headers={'X-Forwarded-For': '198.51.100.4'},
    ):
        assert get_client_ip() == '203.0.113.9'


def test_trusted_proxy_uses_first_forwarded_client_ip(monkeypatch):
    from flask import Flask
    from evalscope.service.rate_limit import get_client_ip

    monkeypatch.setenv('TRUSTED_PROXIES', '127.0.0.1,::1')
    app = Flask(__name__)
    with app.test_request_context(
        '/', environ_base={'REMOTE_ADDR': '127.0.0.1'},
        headers={'X-Forwarded-For': '198.51.100.4, 127.0.0.1'},
    ):
        assert get_client_ip() == '198.51.100.4'


def test_trusted_proxy_ignores_spoofed_leftmost_forwarded_ip(monkeypatch):
    from flask import Flask
    from evalscope.service.rate_limit import get_client_ip

    monkeypatch.setenv('TRUSTED_PROXIES', '127.0.0.1,::1')
    app = Flask(__name__)
    with app.test_request_context(
        '/', environ_base={'REMOTE_ADDR': '127.0.0.1'},
        headers={'X-Forwarded-For': '198.51.100.99, 203.0.113.9'},
    ):
        assert get_client_ip() == '203.0.113.9'


def test_trusted_proxy_ignores_invalid_forwarded_values(monkeypatch):
    from flask import Flask
    from evalscope.service.rate_limit import get_client_ip

    monkeypatch.setenv('TRUSTED_PROXIES', '127.0.0.1,::1')
    app = Flask(__name__)
    with app.test_request_context(
        '/', environ_base={'REMOTE_ADDR': '127.0.0.1'},
        headers={'X-Forwarded-For': 'not-an-ip'},
    ):
        assert get_client_ip() == '127.0.0.1'


def test_sliding_window_returns_retry_after_and_recovers():
    from evalscope.service.rate_limit import SlidingWindowLimiter

    limiter = SlidingWindowLimiter()
    assert limiter.consume('user:1', limit=2, window_seconds=60, now=100.0) == (True, 0)
    assert limiter.consume('user:1', limit=2, window_seconds=60, now=101.0) == (True, 0)
    allowed, retry_after = limiter.consume('user:1', limit=2, window_seconds=60, now=102.0)
    assert allowed is False
    assert retry_after == 58
    assert limiter.consume('user:1', limit=2, window_seconds=60, now=161.0) == (True, 0)


def test_sliding_window_bounds_identity_memory():
    from evalscope.service.rate_limit import SlidingWindowLimiter

    limiter = SlidingWindowLimiter(max_keys=2)
    assert limiter.consume('oldest', limit=5, window_seconds=60, now=100.0)[0]
    assert limiter.consume('newer', limit=5, window_seconds=3600, now=101.0)[0]
    assert limiter.consume('newest', limit=5, window_seconds=60, now=102.0)[0]

    assert len(limiter._events) == 2
    assert 'oldest' not in limiter._events
    assert 'newer' in limiter._events


@pytest.mark.parametrize(
    ('path', 'bucket'),
    [
        ('/api/v1/eval/invoke', 'eval'),
        ('/api/v1/eval/launch', 'eval'),
        ('/api/v1/eval/resume/invoke', 'eval'),
        ('/api/v1/perf/invoke', 'perf'),
        ('/api/v1/perf/launch', 'perf'),
        ('/api/v1/perf/resume/invoke', 'perf'),
        ('/api/v1/aigc/invoke', 'aigc'),
        ('/api/v1/eval/batch/launch', 'batch'),
        ('/api/v1/eval/batch/resume', 'batch'),
        ('/api/v1/perf/batch/launch', 'batch'),
        ('/api/v1/perf/batch/resume', 'batch'),
    ],
)
def test_task_submission_paths_are_classified(path, bucket):
    from evalscope.service.rate_limit import classify_submission_path

    assert classify_submission_path(path, 'POST') == bucket


def test_non_launch_post_is_not_counted_as_task_submission():
    from evalscope.service.rate_limit import classify_submission_path

    assert classify_submission_path('/api/v1/eval/batch/upload', 'POST') is None
    assert classify_submission_path('/api/v1/eval/stop', 'POST') is None
    assert classify_submission_path('/api/v1/reports/delete', 'POST') is None
    assert classify_submission_path('/api/v1/eval/launch', 'GET') is None


def test_registration_rate_limit_is_per_client_ip(app_factory):
    app = app_factory(
        REGISTRATION_MODE='public',
        RATE_LIMIT_REGISTER_REQUESTS='1',
        RATE_LIMIT_REGISTER_WINDOW_SECONDS='3600',
    )
    client = app.test_client()

    first = client.post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123'},
        environ_base={'REMOTE_ADDR': '203.0.113.9'},
    )
    second = client.post(
        '/api/v1/auth/register',
        json={'username': 'bob', 'password': 'secret123'},
        environ_base={'REMOTE_ADDR': '203.0.113.9'},
    )
    other_ip = client.post(
        '/api/v1/auth/register',
        json={'username': 'carol', 'password': 'secret123'},
        environ_base={'REMOTE_ADDR': '203.0.113.10'},
    )

    assert first.status_code == 201
    assert second.status_code == 429
    assert int(second.headers['Retry-After']) > 0
    assert other_ip.status_code == 201


def test_task_rate_limit_is_per_authenticated_user_and_bucket(app_factory):
    app = app_factory(
        REGISTRATION_MODE='public',
        RATE_LIMIT_EVAL_REQUESTS='1',
        RATE_LIMIT_EVAL_WINDOW_SECONDS='3600',
        RATE_LIMIT_PERF_REQUESTS='1',
        RATE_LIMIT_PERF_WINDOW_SECONDS='3600',
    )
    client = app.test_client()
    register = client.post(
        '/api/v1/auth/register',
        json={'username': 'alice', 'password': 'secret123'},
    )
    token = register.get_json()['token']
    headers = {'Authorization': f'Bearer {token}'}

    first_eval = client.post('/api/v1/eval/launch', json={}, headers=headers)
    second_eval = client.post('/api/v1/eval/launch', json={}, headers=headers)
    first_perf = client.post('/api/v1/perf/launch', json={}, headers=headers)

    assert first_eval.status_code == 400
    assert second_eval.status_code == 429
    assert second_eval.get_json()['bucket'] == 'eval'
    assert int(second_eval.headers['Retry-After']) > 0
    assert first_perf.status_code == 400
