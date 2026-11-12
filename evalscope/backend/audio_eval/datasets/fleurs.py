"""Read a prepared, local google/fleurs cmn_hans_cn test export; never download on request.

Expected under the shared cache: audio/fleurs/cmn_hans_cn/test/manifest.json
{"dataset":"google/fleurs","config":"cmn_hans_cn","split":"test",
 "revision":"<pinned upstream revision>","status":"ready",
 "samples":[{"id":"<stable unique ID>","text":"...","audio":"audio/0001.wav"}]}
Audio is decoded ahead of time into local WAV/MP3/OGG/FLAC files by a separate
administrator-controlled preparation step, not by this evaluation endpoint.
"""
import json
import os
import random
import re
import wave
from pathlib import Path
from typing import Any
from urllib.parse import quote

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
FORMATS = {'.wav': b'RIFF', '.mp3': b'ID3', '.flac': b'fLaC', '.ogg': b'OggS'}


def _cache_root() -> Path:
    return Path(os.environ.get('EVALSCOPE_CACHE') or os.environ.get('HF_HOME') or '~/.cache').expanduser().resolve()


def _dataset_dir() -> Path:
    return _cache_root() / 'audio' / 'fleurs' / 'cmn_hans_cn' / 'test'


def _load_manifest() -> tuple[dict[str, Any] | None, str]:
    path = _dataset_dir() / 'manifest.json'
    if not path.is_file():
        return None, 'not_prepared'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None, 'invalid_manifest'
    if (not isinstance(manifest, dict) or manifest.get('dataset') != 'google/fleurs'
            or manifest.get('config') != 'cmn_hans_cn' or manifest.get('split') != 'test'
            or not isinstance(manifest.get('revision'), str) or not manifest['revision'].strip()
            or not isinstance(manifest.get('samples'), list) or not manifest['samples']):
        return None, 'invalid_manifest'
    if manifest.get('status') != 'ready':
        return None, 'not_prepared'
    ids = [s.get('id') for s in manifest['samples'] if isinstance(s, dict)]
    if (len(ids) != len(manifest['samples'])
            or any(not isinstance(sample_id, str) or not ID_RE.fullmatch(sample_id) for sample_id in ids)
            or len(set(ids)) != len(ids)):
        return None, 'invalid_manifest'
    return manifest, 'ready'


def _sample_path(relative_path: str) -> Path:
    if not relative_path or '\x00' in relative_path or Path(relative_path).is_absolute():
        raise ValueError('Invalid FLEURS audio path')
    root = _dataset_dir().resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in FORMATS or not path.is_file():
        raise ValueError('FLEURS audio file is missing or outside the dataset directory')
    return path


def _validate_sample(sample: Any) -> Path:
    if (not isinstance(sample, dict) or not isinstance(sample.get('id'), str)
            or not ID_RE.fullmatch(sample['id']) or not isinstance(sample.get('text'), str)
            or not sample['text'].strip() or not isinstance(sample.get('audio'), str)):
        raise ValueError('Invalid FLEURS sample metadata')
    path = _sample_path(sample['audio'])
    size = path.stat().st_size
    if not 0 < size <= MAX_AUDIO_BYTES:
        raise ValueError('Invalid FLEURS audio size')
    with path.open('rb') as f:
        header = f.read(12)
    suffix = path.suffix.lower()
    if suffix == '.mp3':
        valid = header.startswith(b'ID3') or (len(header) > 2 and header[0] == 0xff and header[1] & 0xe0 == 0xe0)
    elif suffix == '.wav':
        valid = header.startswith(b'RIFF') and header[8:12] == b'WAVE'
    else:
        valid = header.startswith(FORMATS[suffix])
    if not valid:
        raise ValueError('Invalid FLEURS audio encoding')
    if suffix == '.wav':
        try:
            with wave.open(str(path)) as wav:
                if not wav.getnframes() or not wav.getframerate():
                    raise ValueError('Empty WAV audio')
        except (wave.Error, EOFError, OSError) as exc:
            raise ValueError('Corrupt FLEURS WAV audio') from exc
    return path


def _available() -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    manifest, status = _load_manifest()
    if status != 'ready' or manifest is None:
        return None, [], status
    try:
        samples = [(sample, _validate_sample(sample)) for sample in manifest['samples']]
    except (ValueError, OSError, TypeError):
        return manifest, [], 'invalid_samples'
    return manifest, [sample for sample, _ in samples], 'ready'


def get_fleurs_status() -> dict[str, Any]:
    manifest, samples, status = _available()
    return {
        'dataset': 'google/fleurs', 'config': 'cmn_hans_cn', 'split': 'test',
        'ready': status == 'ready', 'status': status,
        'sample_count': len(samples) if status == 'ready' else 0,
        'revision': manifest['revision'] if manifest and status == 'ready' else None,
    }


def list_fleurs_samples(limit: int = 50, offset: int = 0, query: str = '') -> dict[str, Any]:
    manifest, samples, status = _available()
    if status != 'ready':
        return {**get_fleurs_status(), 'samples': [], 'total': 0}
    if query:
        q = query.casefold()
        samples = [s for s in samples if q in s['id'].casefold() or q in s['text'].casefold()]
    return {
        **get_fleurs_status(), 'total': len(samples),
        'samples': [{'id': s['id'], 'text': s['text'],
                     'audio_url': f"/api/v1/audio/fleurs/file/{quote(s['id'], safe='')}"}
                    for s in samples[offset:offset + limit]],
    }


def random_fleurs_sample() -> dict[str, str]:
    _, samples, status = _available()
    if status != 'ready':
        raise FileNotFoundError('FLEURS dataset is not prepared or contains invalid samples')
    sample = random.SystemRandom().choice(samples)
    return {'id': sample['id'], 'text': sample['text'],
            'audio_url': f"/api/v1/audio/fleurs/file/{quote(sample['id'], safe='')}"}


def get_fleurs_sample(sample_id: str) -> dict[str, str]:
    if not isinstance(sample_id, str) or not ID_RE.fullmatch(sample_id):
        raise FileNotFoundError('FLEURS sample not found')
    manifest, samples, status = _available()
    if status != 'ready' or manifest is None:
        raise FileNotFoundError('FLEURS dataset is not prepared or contains invalid samples')
    for sample in samples:
        if sample['id'] == sample_id:
            return {'id': sample_id, 'reference_text': sample['text'],
                    'audio_path': str(_validate_sample(sample)), 'revision': manifest['revision']}
    raise FileNotFoundError('FLEURS sample not found')
