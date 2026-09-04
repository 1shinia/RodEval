"""Authentication blueprint — register, login, JWT management."""

import datetime
import os
import secrets
import sqlite3
import threading
import time
import uuid

import jwt
from flask import Blueprint, current_app, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import _get_conn, _write

logger = __import__('evalscope.utils.logger', fromlist=['get_logger']).get_logger()

bp_auth = Blueprint('auth', __name__, url_prefix='/api/v1/auth')

# Default admin username (bootstrap only — see ensure_admin_user below)
_DEFAULT_ADMIN = 'admin'

# JWT secret. Resolution order:
#   1. JWT_SECRET env var (explicit deployment config)
#   2. persisted random secret in {OUTPUT_DIR}/.jwt_secret (created on first
#      import with 0600 perms; survives restarts so tokens stay valid, and is
#      NOT in the DB so a meta.db restore doesn't drag auth state along)
# The old hardcoded 'rod-eval-jwt-secret-2024' is gone: existing tokens are
# invalidated once on upgrade (users just log in again).
_JWT_EXPIRY_HOURS = int(os.environ.get('JWT_EXPIRY_HOURS', '72'))
_AUTH_COOKIE = 'evalscope_auth'


def _load_or_create_jwt_secret() -> str:
    env = os.environ.get('JWT_SECRET')
    if env:
        return env
    try:
        from ..utils.log import OUTPUT_DIR as _OUTPUT_DIR
        secret_path = os.path.join(str(_OUTPUT_DIR), '.jwt_secret')
    except Exception:
        secret_path = os.path.join(os.getcwd(), 'outputs', '.jwt_secret')
    try:
        if os.path.isfile(secret_path):
            with open(secret_path) as f:
                secret = f.read().strip()
            if secret:
                return secret
        os.makedirs(os.path.dirname(secret_path), exist_ok=True)
        import secrets as _secrets
        secret = _secrets.token_hex(32)
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(secret)
        return secret
    except Exception:
        # Last resort: ephemeral per-process secret (all sessions invalidate
        # on restart, but the service still works).
        import secrets as _secrets
        return _secrets.token_hex(32)


_JWT_SECRET = _load_or_create_jwt_secret()


def ensure_admin_user() -> None:
    """Ensure there is at least one active admin account.

    Safe bootstrap after meta.db loss (previously nobody could log in).
    Password source: EVALSCOPE_ADMIN_PASSWORD env var; if unset, a random
    one is generated and printed ONCE to the service log.
    Never resets an existing *active* admin's password.  If soft deletion left
    the system with no active admin and the default ``admin`` username still
    exists, that row is reactivated instead of attempting a conflicting INSERT.
    """
    conn = _get_conn()
    row = conn.execute("SELECT id FROM users WHERE role = 'admin' AND deleted_at IS NULL LIMIT 1").fetchone()
    if row is not None:
        return

    default_row = conn.execute(
        'SELECT id FROM users WHERE username = ? LIMIT 1', (_DEFAULT_ADMIN,)
    ).fetchone()

    password = os.environ.get('EVALSCOPE_ADMIN_PASSWORD', '')
    generated = False
    if not password:
        import secrets as _secrets
        password = _secrets.token_urlsafe(12)
        generated = True

    pw_hash = generate_password_hash(password)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _op(c) -> None:
        if default_row is not None:
            c.execute(
                '''UPDATE users
                   SET password_hash = ?, role = 'admin', deleted_at = NULL
                   WHERE id = ?''',
                (pw_hash, default_row['id']),
            )
        else:
            c.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
                (_DEFAULT_ADMIN, pw_hash, now),
            )

    try:
        _write(_op)
    except Exception as e:
        logger.warning(f'Admin bootstrap failed (non-fatal): {e}')
        return
    action = 'Reactivated' if default_row is not None else 'Bootstrapped'
    if generated:
        logger.warning(
            f'{action} admin user "{_DEFAULT_ADMIN}" with random password: {password} '
            '(shown once — change it after first login, or set EVALSCOPE_ADMIN_PASSWORD)'
        )
    else:
        logger.info(f'{action} admin user "{_DEFAULT_ADMIN}" from EVALSCOPE_ADMIN_PASSWORD')


def _user_by_username(username: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        'SELECT id, username, password_hash, role FROM users WHERE username = ? AND deleted_at IS NULL',
        (username,),
    ).fetchone()
    return dict(row) if row else None


def _create_token(user: dict) -> str:
    payload = {
        'jti': uuid.uuid4().hex,
        'sub': str(user['id']),
        'username': user['username'],
        'role': user['role'],
        'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=_JWT_EXPIRY_HOURS),
        'iat': datetime.datetime.now(datetime.timezone.utc),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm='HS256')


# Short-TTL cache for token blacklist lookups; a logout writes the entry into
# the cache immediately (see _blacklist_token) so revocation takes effect at
# once within this process, while the TTL bounds staleness across processes.
_blacklist_cache: dict[str, tuple[float, bool]] = {}
_blacklist_cache_ttl = float(os.environ.get('AUTH_BLACKLIST_CACHE_TTL', '30'))
_blacklist_cache_lock = threading.Lock()


def _is_blacklisted(jti: str) -> bool:
    """Return True if the token jti is in the blacklist (short-TTL cached)."""
    now = time.monotonic()
    with _blacklist_cache_lock:
        cached = _blacklist_cache.get(jti)
        if cached is not None and cached[0] > now:
            return cached[1]
    conn = _get_conn()
    row = conn.execute('SELECT 1 FROM token_blacklist WHERE jti = ?', (jti,)).fetchone()
    result = row is not None
    with _blacklist_cache_lock:
        _blacklist_cache[jti] = (now + _blacklist_cache_ttl, result)
    return result


def _blacklist_token(jti: str, exp: float) -> None:
    """Add a token jti to the blacklist with its expiry for cleanup."""
    exp_str = datetime.datetime.fromtimestamp(exp, datetime.timezone.utc).isoformat()
    _write(lambda conn: conn.execute(
        'INSERT OR IGNORE INTO token_blacklist (jti, expires_at) VALUES (?, ?)', (jti, exp_str)))
    # Refresh the cache so revocation takes effect immediately on this process.
    with _blacklist_cache_lock:
        _blacklist_cache[jti] = (time.monotonic() + _blacklist_cache_ttl, True)


def _cleanup_expired_tokens() -> None:
    """Remove expired tokens from the blacklist."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write(lambda conn: conn.execute('DELETE FROM token_blacklist WHERE expires_at < ?', (now,)))


def verify_token(token: str) -> dict | None:
    """Return user dict if token is valid, None otherwise."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def _extract_request_token(*, allow_cookie: bool = True) -> str | None:
    """Read auth from Bearer header, optionally falling back to a cookie.

    Cookie fallback is intended for safe browser GET transports such as
    EventSource/iframe/download.  State-changing API calls still require an
    explicit Bearer token, avoiding a broad cookie-auth CSRF surface.
    """
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    if allow_cookie:
        return request.cookies.get(_AUTH_COOKIE)
    return None


def _set_auth_cookie(response, token: str):
    response.set_cookie(
        _AUTH_COOKIE,
        token,
        max_age=_JWT_EXPIRY_HOURS * 60 * 60,
        httponly=True,
        secure=bool(request.is_secure),
        samesite='Lax',
        path='/',
    )
    return response


# Process-local cache for the per-request user lookup in require_auth.  A short
# TTL keeps soft-delete / role changes visible within seconds while avoiding a
# SQLite read on every polled request.  Invalidated explicitly on user mutation.
_user_cache: dict[int, tuple[float, dict | None]] = {}
_user_cache_ttl = float(os.environ.get('AUTH_USER_CACHE_TTL', '30'))
_user_cache_lock = threading.Lock()


def _get_user_row(uid: int) -> dict | None:
    """Return the active user row (dict) for *uid*, cached for a short TTL."""
    now = time.monotonic()
    with _user_cache_lock:
        cached = _user_cache.get(uid)
        if cached is not None and cached[0] > now:
            return cached[1]
    row = _get_conn().execute(
        'SELECT id, username, role FROM users WHERE id = ? AND deleted_at IS NULL', (uid,)
    ).fetchone()
    result = dict(row) if row else None
    with _user_cache_lock:
        _user_cache[uid] = (now + _user_cache_ttl, result)
    return result


def require_auth():
    """Flask before_request hook — require authenticated access by default.

    Browser-only transports such as EventSource and iframe no longer bypass
    authentication: they use the HttpOnly auth cookie established at login.
    """
    if request.method == 'OPTIONS':
        return None

    # Only genuinely non-user-specific discovery/bootstrap endpoints are public.
    for prefix in (
        '/health', '/dashboard', '/api/v1/auth/', '/api/v1/config',
        '/api/v1/benchmarks', '/api/v1/eval/benchmarks',
        '/api/v1/perf/template', '/api/v1/eval/batch/template',
        '/api/v1/perf/batch/template',
    ):
        if request.path.startswith(prefix):
            return None

    token = _extract_request_token(allow_cookie=request.method in ('GET', 'HEAD'))
    if not token:
        return jsonify({'error': 'Missing or invalid token'}), 401

    user = verify_token(token)
    if user is None:
        return jsonify({'error': 'Token expired or invalid'}), 401
    if _is_blacklisted(user.get('jti', '')):
        return jsonify({'error': 'Token has been revoked'}), 401

    try:
        uid = int(user.get('sub', '0'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Token user is invalid'}), 401
    row = _get_user_row(uid)
    if row is None:
        return jsonify({'error': 'User account is disabled or deleted'}), 401
    user['username'] = row['username']
    user['role'] = row['role']
    request.current_user = user
    return None


def get_current_user_id() -> int | None:
    """Return the current authenticated user_id, or None when unauthenticated.

    Fail-closed: public endpoints must not silently assume an admin identity.
    Authenticated routes always get a real id because require_auth populates
    ``request.current_user`` before dispatch.
    """
    user = getattr(request, 'current_user', None)
    return int(user['sub']) if user else None


def get_user_output_dir(user_id: int = None) -> str:
    """Return user-specific output directory. (P3: deferred — not yet active)"""
    from ..utils import OUTPUT_DIR
    uid = user_id if user_id is not None else get_current_user_id()
    return os.path.join(OUTPUT_DIR, str(uid))


def get_current_role() -> str:
    """Return the current user's role ('admin'/'user'); '' when unauthenticated."""
    user = getattr(request, 'current_user', None)
    return (user or {}).get('role', '')


def check_task_ownership(table: str, task_id: str) -> tuple[bool, int | None]:
    """Return ``(allowed, owner_user_id)`` for a task in *table*.

    Rules:
      - Row exists and belongs to the requester  -> allowed
      - Row exists, belongs to someone else      -> denied
      - No row (unindexed/legacy directory)      -> admin only
    """
    from ..db import _TASK_OWNERSHIP_TABLES, _get_conn
    if table not in _TASK_OWNERSHIP_TABLES:
        raise ValueError(f'Unsupported task table: {table}')
    uid = get_current_user_id()
    row = _get_conn().execute(
        f'SELECT user_id FROM {table} WHERE task_id = ?', (task_id,)
    ).fetchone()
    if row is None:
        return get_current_role() == 'admin', None
    return row[0] == uid, row[0]


def check_task_artifact_access(task_id: str, tables: tuple[str, ...]) -> bool:
    """Authorize reports/logs/SSE/files through the shared task policy."""
    uid = get_current_user_id()
    if uid is None:
        # Unauthenticated requests must never be treated as admin (fail-closed).
        return False
    from ..task_access import task_artifact_owned_by
    from ..utils import OUTPUT_DIR
    return task_artifact_owned_by(
        task_id,
        tables,
        user_id=uid,
        is_admin=get_current_role() == 'admin',
        output_dir=str(OUTPUT_DIR),
    )


@bp_auth.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()

    if not username or not password:
        return jsonify({'error': 'username and password are required'}), 400
    if len(username) < 2 or len(username) > 32:
        return jsonify({'error': 'username must be 2-32 characters'}), 400
    if len(password) < 6:
        return jsonify({'error': 'password must be at least 6 characters'}), 400

    conn = _get_conn()
    existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if existing:
        return jsonify({'error': '用户名已存在'}), 409

    pw_hash = generate_password_hash(password)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        _write(lambda conn: conn.execute(
            'INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
            (username, pw_hash, 'user', now),
        ))
    except sqlite3.IntegrityError:
        # The UNIQUE constraint is the cross-process authority.  A concurrent
        # registration can pass the optimistic pre-check above, then lose the
        # INSERT race; report that as a normal conflict instead of HTTP 500.
        # Do not mask unrelated integrity failures.
        if _user_by_username(username):
            return jsonify({'error': '用户名已存在'}), 409
        raise

    user = _user_by_username(username)
    if not user:
        return jsonify({'error': 'Registration failed'}), 500
    token = _create_token(user)
    response = jsonify({
        'token': token,
        'user': {'id': user['id'], 'username': user['username'], 'role': user['role']},
    })
    return _set_auth_cookie(response, token), 201


@bp_auth.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()

    if not username or not password:
        return jsonify({'error': 'username and password are required'}), 400

    user = _user_by_username(username)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': '用户名或密码错误'}), 401

    token = _create_token(user)
    response = jsonify({
        'token': token,
        'user': {'id': user['id'], 'username': user['username'], 'role': user['role']},
    })
    return _set_auth_cookie(response, token)


@bp_auth.route('/logout', methods=['POST'])
def logout():
    """Invalidate the current token by adding its jti to the blacklist."""
    token = _extract_request_token()
    if not token:
        return jsonify({'error': 'Missing token'}), 400
    payload = verify_token(token)
    if payload is None:
        return jsonify({'error': 'Token invalid or already expired'}), 400
    _blacklist_token(payload['jti'], payload['exp'])
    _cleanup_expired_tokens()  # opportunistic cleanup
    response = jsonify({'ok': True})
    response.delete_cookie(_AUTH_COOKIE, path='/')
    return response, 200


@bp_auth.route('/password', methods=['PUT'])
def change_password():
    """Change current user's password."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing token'}), 400
    payload = verify_token(auth_header[7:])
    if payload is None:
        return jsonify({'error': 'Token invalid or expired'}), 401
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    old_password = (data.get('old_password') or '').strip()
    new_password = (data.get('new_password') or '').strip()
    if not old_password or not new_password:
        return jsonify({'error': 'old_password and new_password are required'}), 400
    if len(new_password) < 6:
        return jsonify({'error': '新密码至少6个字符'}), 400
    user = _user_by_username(payload['username'])
    if not user or not check_password_hash(user['password_hash'], old_password):
        return jsonify({'error': '当前密码错误'}), 401
    _write(lambda conn: conn.execute(
        'UPDATE users SET password_hash = ? WHERE id = ?',
        (generate_password_hash(new_password), user['id'])))
    return jsonify({'ok': True}), 200


def _require_admin() -> dict | None:
    """Return an active admin account for a valid, non-revoked token."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    payload = verify_token(auth_header[7:])
    if payload is None or payload.get('role') != 'admin' or _is_blacklisted(payload.get('jti', '')):
        return None
    try:
        uid = int(payload.get('sub', '0'))
    except (TypeError, ValueError):
        return None
    row = _get_conn().execute(
        "SELECT id, username, role FROM users WHERE id = ? AND role = 'admin' AND deleted_at IS NULL",
        (uid,),
    ).fetchone()
    if row is None:
        return None
    payload['username'] = row['username']
    payload['role'] = row['role']
    return payload


@bp_auth.route('/users', methods=['GET'])
def list_users():
    """List all users (admin only)."""
    admin = _require_admin()
    if admin is None:
        return jsonify({'error': 'Admin access required'}), 403
    conn = _get_conn()
    rows = conn.execute('SELECT id, username, role, created_at FROM users WHERE deleted_at IS NULL ORDER BY id').fetchall()
    return jsonify({'users': [dict(r) for r in rows]}), 200


@bp_auth.route('/users', methods=['POST'])
def create_user():
    """Create a new user (admin only)."""
    admin = _require_admin()
    if admin is None:
        return jsonify({'error': 'Admin access required'}), 403
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    role = (data.get('role') or 'user').strip()
    if not username or not password:
        return jsonify({'error': 'username and password are required'}), 400
    if len(username) < 2 or len(username) > 32:
        return jsonify({'error': 'username must be 2-32 characters'}), 400
    if len(password) < 6:
        return jsonify({'error': 'password must be at least 6 characters'}), 400
    conn = _get_conn()
    existing = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if existing:
        return jsonify({'error': '用户名已存在'}), 409
    pw_hash = generate_password_hash(password)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write(lambda conn: conn.execute(
        'INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
        (username, pw_hash, role, now),
    ))
    new_user = _user_by_username(username)
    return jsonify({'user': {'id': new_user['id'], 'username': new_user['username'], 'role': new_user['role']}}), 201


@bp_auth.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id: int):
    """Delete a user (admin only, cannot delete self)."""
    admin = _require_admin()
    if admin is None:
        return jsonify({'error': 'Admin access required'}), 403
    if int(admin['sub']) == user_id:
        return jsonify({'error': '不能删除自己'}), 400
    deleted_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    removed = _write(lambda conn: conn.execute(
        'UPDATE users SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL',
        (deleted_at, user_id),
    ).rowcount)
    if removed == 0:
        return jsonify({'error': '用户不存在'}), 404
    with _user_cache_lock:
        _user_cache.pop(user_id, None)
    return jsonify({'ok': True}), 200


@bp_auth.route('/users/<int:user_id>/password', methods=['PUT'])
def reset_password(user_id: int):
    """Reset a user's password (admin only)."""
    admin = _require_admin()
    if admin is None:
        return jsonify({'error': 'Admin access required'}), 403
    data = request.get_json()
    if not data or not data.get('password'):
        return jsonify({'error': 'password is required'}), 400
    password = data['password'].strip()
    if len(password) < 6:
        return jsonify({'error': 'password must be at least 6 characters'}), 400
    pw_hash = generate_password_hash(password)
    updated = _write(lambda conn: conn.execute(
        'UPDATE users SET password_hash = ? WHERE id = ? AND deleted_at IS NULL', (pw_hash, user_id)).rowcount)
    if updated == 0:
        return jsonify({'error': '用户不存在'}), 404
    return jsonify({'ok': True}), 200


_RESET_TOKEN_TTL_HOURS = int(os.environ.get('RESET_TOKEN_TTL_HOURS', '24'))


@bp_auth.route('/users/<int:user_id>/reset-token', methods=['POST'])
def create_reset_token(user_id: int):
    """Generate a one-time password reset token for a user (admin only).

    Invalidate any previous unused tokens for the user so only the newest one
    is valid; also opportunistically purge expired tokens.
    """
    admin = _require_admin()
    if admin is None:
        return jsonify({'error': 'Admin access required'}), 403
    conn = _get_conn()
    user = conn.execute(
        'SELECT id, username FROM users WHERE id = ? AND deleted_at IS NULL', (user_id,)
    ).fetchone()
    if user is None:
        return jsonify({'error': '用户不存在'}), 404

    token = secrets.token_urlsafe(32)
    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = (now + datetime.timedelta(hours=_RESET_TOKEN_TTL_HOURS)).isoformat()

    def _op(c):
        c.execute('DELETE FROM password_reset_tokens WHERE expires_at < ?', (now.isoformat(),))
        c.execute('DELETE FROM password_reset_tokens WHERE user_id = ?', (user_id,))
        c.execute(
            'INSERT INTO password_reset_tokens (token, user_id, expires_at, used, created_at) '
            'VALUES (?, ?, ?, 0, ?)',
            (token, user_id, expires_at, now.isoformat()),
        )
    _write(_op)

    return jsonify({'token': token, 'expires_at': expires_at}), 201


@bp_auth.route('/reset-password', methods=['POST'])
def reset_password_with_token():
    """Reset a user's password using a one-time reset token (public).

    The token must be unused and unexpired, and the target user must still be
    active.  Every invalid-token case returns the same message so this endpoint
    does not leak whether a token or user exists.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    token = (data.get('token') or '').strip()
    password = (data.get('password') or '').strip()
    if not token or not password:
        return jsonify({'error': 'token and password are required'}), 400
    if len(password) < 6:
        return jsonify({'error': '新密码至少6个字符'}), 400

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = _get_conn()
    row = conn.execute(
        'SELECT user_id FROM password_reset_tokens '
        'WHERE token = ? AND used = 0 AND expires_at > ?',
        (token, now),
    ).fetchone()
    if row is None:
        return jsonify({'error': '链接无效或已过期'}), 400
    user_id = row['user_id']
    user = conn.execute(
        'SELECT id FROM users WHERE id = ? AND deleted_at IS NULL', (user_id,)
    ).fetchone()
    if user is None:
        return jsonify({'error': '链接无效或已过期'}), 400

    pw_hash = generate_password_hash(password)

    def _op(c):
        c.execute('UPDATE users SET password_hash = ? WHERE id = ?', (pw_hash, user_id))
        c.execute('UPDATE password_reset_tokens SET used = 1 WHERE token = ?', (token,))
    _write(_op)

    with _user_cache_lock:
        _user_cache.pop(user_id, None)
    return jsonify({'ok': True}), 200
