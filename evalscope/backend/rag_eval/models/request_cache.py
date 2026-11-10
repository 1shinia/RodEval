"""Atomic, non-pickle request checkpoints (only digests and numeric arrays on disk)."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from evalscope.utils.logger import get_logger

logger = get_logger()


def request_cache_path(model, kind, request):
    """Return an isolated digest path, or disable caching for unsupported inputs.

    Credentials/headers must never be included in request. Neither namespace nor
    request JSON is persisted: they only contribute to SHA-256 identities.
    """
    directory = getattr(model, 'request_cache_dir', None)
    namespace = getattr(model, 'request_cache_namespace', None)
    if not directory or not isinstance(namespace, str):
        return None
    try:
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                             allow_nan=False).encode('utf-8')
    except (TypeError, ValueError):
        return None
    scope = hashlib.sha256(namespace.encode('utf-8')).hexdigest()
    digest = hashlib.sha256(kind.encode('ascii') + b'\0v1\0' + encoded).hexdigest()
    return Path(directory) / scope / (digest + '.npz')


def load_response(path, count, ndim):
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = archive['values']
            if (values.dtype.kind not in 'fiu' or values.ndim != ndim or len(values) != count
                    or not np.isfinite(values).all() or (ndim == 2 and values.shape[1] == 0)):
                return None
            return values.tolist()
    except Exception:
        # Interrupted writes, corrupt archives and incompatible versions are misses.
        return None


def save_response(path, response):
    """Commit each completed request before the caller starts the next one.

    Cache I/O is best effort and must not change provider response semantics.
    """
    if path is None:
        return
    temporary = None
    try:
        values = np.asarray(response)
        if values.dtype.kind not in 'fiu':
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.request-', suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, values=values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        # Exceptions and paths may contain provider input or credentials.
        logger.warning('Request cache write failed; response will be recomputed on resume.')
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning('Request cache temporary cleanup failed.')
