"""Isolated server-side administrator recovery tests."""
import importlib
import sqlite3

import jwt
import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

from evalscope.service import db
from evalscope.service.blueprints import auth


@pytest.fixture()
def recovery(tmp_path, monkeypatch):
    db.init_db(str(tmp_path))
    conn = db._get_conn()
    conn.execute('DELETE FROM users')
    for name, role in [('admin', 'admin'), ('alice', 'user')]:
        conn.execute('INSERT INTO users(username,password_hash,role,created_at) VALUES (?,?,?,?)',
                     (name, generate_password_hash('oldpass123'), role, '2026-01-01'))
    conn.execute("INSERT INTO eval_reports(task_id,model_name,dataset_name) VALUES ('keep','model','data')")
    conn.commit()
    auth._user_cache.clear()
    auth._blacklist_cache.clear()
    auth._login_failures.clear()
    auth._login_locks.clear()
    app = Flask(__name__)
    app.register_blueprint(auth.bp_auth)
    app.before_request(auth.require_auth)
    module = importlib.import_module('evalscope.service.reset_admin_password')
    monkeypatch.setattr(module.sys.stdin, 'isatty', lambda: True)
    answers = iter(['newpass456', 'newpass456'])
    monkeypatch.setattr(module.getpass, 'getpass', lambda prompt: next(answers))
    path = tmp_path / 'evalscope_meta.db'
    yield module, path, conn, app


def snapshot(conn):
    return list(conn.iterdump())


def test_recovery_revokes_only_target_sessions_and_links(recovery):
    module, path, conn, app = recovery
    def login(name, password):
        return app.test_client().post('/api/v1/auth/login', json={'username': name, 'password': password})
    old = login('admin', 'oldpass123').get_json()['token']
    other = login('alice', 'oldpass123').get_json()['token']
    headers = {'Authorization': f'Bearer {old}'}
    assert app.test_client().get('/api/v1/auth/users', headers=headers).status_code == 200
    uid = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()[0]
    link = app.test_client().post(f'/api/v1/auth/users/{uid}/reset-token', headers=headers).get_json()['token']
    legacy = jwt.decode(old, auth._JWT_SECRET, algorithms=['HS256'])
    legacy.pop('auth_version', None)
    legacy_token = jwt.encode(legacy, auth._JWT_SECRET, algorithm='HS256')
    assert module.main(['--database', str(path), '--username', 'admin']) == 0
    assert login('admin', 'oldpass123').status_code == 401
    fresh = login('admin', 'newpass456')
    assert fresh.status_code == 200
    assert jwt.decode(fresh.get_json()['token'], auth._JWT_SECRET, algorithms=['HS256'])['auth_version'] == 1
    for token in (old, legacy_token):
        assert app.test_client().get('/api/v1/auth/users', headers={'Authorization': f'Bearer {token}'}).status_code == 401
    assert app.test_client().get('/api/v1/auth/users', headers={'Authorization': f'Bearer {other}'}).status_code == 403
    assert login('alice', 'oldpass123').status_code == 200
    assert app.test_client().post('/api/v1/auth/reset-password', json={'token': link, 'password': 'another123'}).status_code == 400
    assert conn.execute("SELECT task_id FROM eval_reports WHERE task_id='keep'").fetchone()


@pytest.mark.parametrize('target', ['missing', 'alice', 'deleted'])
def test_invalid_target_does_not_prompt_or_write(recovery, monkeypatch, target):
    module, path, conn, _ = recovery
    if target == 'deleted':
        conn.execute("UPDATE users SET deleted_at='2026-01-01' WHERE username='admin'")
        conn.commit()
        target = 'admin'
    before = snapshot(conn)
    monkeypatch.setattr(module.getpass, 'getpass', lambda _: pytest.fail('must validate before prompting'))
    assert module.main(['--database', str(path), '--username', target]) == 1
    assert snapshot(conn) == before


@pytest.mark.parametrize('answers', [('newpass456', 'different'), ('short', 'short'), ('      ', '      ')])
def test_bad_password_does_not_write(recovery, monkeypatch, answers):
    module, path, conn, _ = recovery
    before = snapshot(conn)
    values = iter(answers)
    monkeypatch.setattr(module.getpass, 'getpass', lambda _: next(values))
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before


def test_wrong_path_and_schema_do_not_create_or_initialize(recovery, tmp_path, monkeypatch):
    module, _, _, _ = recovery
    monkeypatch.setattr(module.getpass, 'getpass', lambda _: pytest.fail('must validate before prompting'))
    missing = tmp_path / 'missing.db'
    assert module.main(['--database', str(missing), '--username', 'admin']) == 1
    assert not missing.exists()
    empty = tmp_path / 'empty.db'
    empty.touch()
    assert module.main(['--database', str(empty), '--username', 'admin']) == 1
    assert empty.read_bytes() == b''


@pytest.mark.parametrize('exception', [KeyboardInterrupt, EOFError])
def test_cancel_does_not_write(recovery, monkeypatch, exception):
    module, path, conn, _ = recovery
    before = snapshot(conn)
    def cancel(_):
        raise exception()
    monkeypatch.setattr(module.getpass, 'getpass', cancel)
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before


def test_noninteractive_rejected(recovery, monkeypatch):
    module, path, conn, _ = recovery
    before = snapshot(conn)
    monkeypatch.setattr(module.sys.stdin, 'isatty', lambda: False)
    monkeypatch.setattr(module.getpass, 'getpass', lambda _: pytest.fail('no stdin fallback'))
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before


@pytest.mark.parametrize('arguments', [[], ['--username', 'admin'],
                                      ['--database', '/tmp/missing.db'],
                                      ['--database', '/tmp/missing.db', '--username', 'admin', '--password', 'forbidden']])
def test_required_arguments_and_no_password_option(recovery, arguments):
    module, _, conn, _ = recovery
    before = snapshot(conn)
    with pytest.raises(SystemExit) as error:
        module.main(arguments)
    assert error.value.code == 2
    assert snapshot(conn) == before


def test_no_active_admin_does_not_bootstrap(recovery, monkeypatch):
    module, path, conn, _ = recovery
    conn.execute("UPDATE users SET role='user'")
    conn.commit()
    before = snapshot(conn)
    monkeypatch.setattr(module.getpass, 'getpass', lambda _: pytest.fail('must validate before prompting'))
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before


def test_getpass_warning_cannot_fall_back(recovery, monkeypatch):
    import warnings
    module, path, conn, _ = recovery
    before = snapshot(conn)
    def fallback(_):
        warnings.warn('terminal unavailable', module.getpass.GetPassWarning)
        pytest.fail('must not read stdin after warning')
    monkeypatch.setattr(module.getpass, 'getpass', fallback)
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before


@pytest.mark.parametrize('ddl', ['ALTER TABLE users RENAME COLUMN role TO legacy_role',
                               'ALTER TABLE users RENAME COLUMN auth_version TO legacy_version',
                               'UPDATE schema_version SET version=99 WHERE version=28'])
def test_incompatible_schema_does_not_prompt_or_write(recovery, monkeypatch, ddl):
    module, path, conn, _ = recovery
    conn.execute(ddl)
    conn.commit()
    before = snapshot(conn)
    monkeypatch.setattr(module.getpass, 'getpass', lambda _: pytest.fail('must validate before prompting'))
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before


def test_v27_upgrade_preserves_accounts_and_legacy_sessions(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(db, 'SCHEMA_VERSION', 27)
        db.init_db(str(tmp_path))
    conn = db._get_conn()
    conn.execute("INSERT INTO users(username,password_hash,role,created_at) VALUES ('prior','hash','admin','2026')")
    conn.commit()
    db.init_db(str(tmp_path))
    conn = db._get_conn()
    row = conn.execute("SELECT * FROM users WHERE username='prior'").fetchone()
    assert row['password_hash'] == 'hash'
    assert row['auth_version'] == 0
    assert conn.execute('SELECT MAX(version) FROM schema_version').fetchone()[0] == 28
    payload = jwt.decode(auth._create_token(dict(row)), auth._JWT_SECRET, algorithms=['HS256'])
    del payload['auth_version']
    token = jwt.encode(payload, auth._JWT_SECRET, algorithm='HS256')
    app = Flask(__name__)
    auth._user_cache.clear()
    with app.test_request_context('/private', headers={'Authorization': f'Bearer {token}'}):
        assert auth.require_auth() is None


def test_delete_failure_rolls_back_password_and_version(recovery):
    module, path, conn, _ = recovery
    conn.execute("CREATE TRIGGER prevent_delete BEFORE DELETE ON password_reset_tokens BEGIN SELECT RAISE(ABORT,'probe'); END")
    uid = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()[0]
    conn.execute("INSERT INTO password_reset_tokens VALUES ('probe',?,'2099',0,'2026')", (uid,))
    conn.commit()
    before = snapshot(conn)
    assert module.main(['--database', str(path), '--username', 'admin']) == 1
    assert snapshot(conn) == before
