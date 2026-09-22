"""Shared, credential-free protocol helpers for resumable batch jobs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_SENSITIVE_KEYS = {
    'api_key',
    'apikey',
    'authorization',
    'proxy_authorization',
    'cookie',
    'set_cookie',
    'password',
    'secret',
    'token',
    'access_token',
    'refresh_token',
    'x_api_key',
    'x_auth_token',
}


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace('-', '_')
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith('_api_key')
        or normalized.endswith('_token')
        or normalized.endswith('_password')
        or normalized.endswith('_secret')
        or normalized.endswith('_secret_key')
    )


def _without_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_credentials(item)
            for key, item in value.items()
            if not _is_sensitive_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_credentials(item) for item in value]
    return value


def batch_manifest_hash(rows: list[dict[str, Any]], shared_config: dict[str, Any]) -> str:
    """Hash ordered execution inputs while excluding every credential field."""
    payload = {
        'rows': _without_credentials(rows),
        'shared_config': _without_credentials(shared_config),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def resumable_row_indexes(items: list[dict[str, Any]]) -> set[int]:
    """Return stable row indexes that have not reached a durable terminal result."""
    return {
        int(item['row_index'])
        for item in items
        if item.get('status') in {'pending', 'interrupted'}
    }
