"""Authentication blueprint — register, login, JWT management."""

import datetime
import os
import sqlite3

import jwt
from flask import Blueprint, current_app, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import _get_conn

logger = __import__('evalscope.utils.logger', fromlist=['get_logger']).get_logger()

bp_auth = Blueprint('auth', __name__, url_prefix='/api/v1/auth')

# Default admin password on first run
_DEFAULT_ADMIN = 'admin'
_DEFAULT_PASSWORD = 'admin123'

# JWT secret — fixed value (override with JWT_SECRET env var)
_JWT_SECRET = os.environ.get('JWT_SECRET', 'rod-eval-jwt-secret-2024')
_JWT_EXPIRY_HOURS = int(os.environ.get('JWT_EXPIRY_HOURS', '72'))


def _user_by_username(username: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        'SELECT id, username, password_hash, role FROM users WHERE username = ?',
        (username,),
    ).fetchone()
    return dict(row) if row else None


def _create_token(user: dict) -> str:
    payload = {
        'sub': str(user['id']),
        'username': user['username'],
        'role': user['role'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=_JWT_EXPIRY_HOURS),
        'iat': datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm='HS256')


def verify_token(token: str) -> dict | None:
    """Return user dict if token is valid, None otherwise."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth():
    """Flask before_request hook — skip auth for public paths, require Bearer token otherwise."""
    if request.method == 'OPTIONS':
        return None

    # Public path prefixes (no auth required)
    for prefix in ('/health', '/dashboard', '/api/v1/auth/',
                   '/api/v1/config', '/api/v1/benchmarks',
                   '/api/v1/eval/benchmarks', '/api/v1/perf/template',
                   '/api/v1/eval/batch/template', '/api/v1/perf/batch/template'):
        if request.path.startswith(prefix):
            return None

    # Endpoints that can't send auth headers (EventSource / iframe / window.open)
    for suffix in ('/log/stream', '/progress/stream', '/report', '/chart', '/html'):
        if request.path.endswith(suffix):
            return None

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid token'}), 401

    token = auth_header[7:]
    user = verify_token(token)
    if user is None:
        return jsonify({'error': 'Token expired or invalid'}), 401

    request.current_user = user
    return None


def get_current_user_id() -> int:
    """Return the current authenticated user_id (1 = admin default for public endpoints)."""
    user = getattr(request, 'current_user', None)
    return int(user['sub']) if user else 1


def get_user_output_dir(user_id: int = None) -> str:
    """Return user-specific output directory."""
    from ..utils import OUTPUT_DIR
    uid = user_id if user_id is not None else get_current_user_id()
    return os.path.join(OUTPUT_DIR, str(uid))


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
    now = datetime.datetime.utcnow().isoformat()
    conn.execute(
        'INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
        (username, pw_hash, 'user', now),
    )
    conn.commit()

    user = _user_by_username(username)
    if not user:
        return jsonify({'error': 'Registration failed'}), 500
    token = _create_token(user)
    return jsonify({
        'token': token,
        'user': {'id': user['id'], 'username': user['username'], 'role': user['role']},
    }), 201


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
    return jsonify({
        'token': token,
        'user': {'id': user['id'], 'username': user['username'], 'role': user['role']},
    })
