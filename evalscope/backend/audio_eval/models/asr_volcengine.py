"""Volcengine native flash recording-file ASR (Base64 upload)."""
import base64
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

from .base import AudioModelBase


RESOURCE_ID = 'volc.bigasr.auc_turbo'
FORMATS = {'.wav': 'wav', '.mp3': 'mp3', '.ogg': 'ogg'}
LANGUAGES = {'zh': 'zh-CN', 'en': 'en-US'}


def _speech_base(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Volcengine API URL must be an HTTPS base URL')
    path = parsed.path.rstrip('/')
    if path not in ('', '/api/v3'):
        raise ValueError('Volcengine API URL must be the host or /api/v3 base')
    return f'{parsed.scheme}://{parsed.netloc}'


class AsrModelVolcengine(AudioModelBase):
    """Synchronous native flash ASR; the standard submit/query API requires a public URL."""

    def generate(self, audio_path: str, **kwargs):
        if self.model_name != RESOURCE_ID:
            raise ValueError(f'Volcengine flash ASR model must be {RESOURCE_ID}')
        path = Path(audio_path)
        fmt = FORMATS.get(path.suffix.lower())
        if not fmt:
            raise ValueError('Volcengine flash ASR supports WAV, MP3 or OGG only')
        if not path.is_file():
            raise FileNotFoundError(audio_path)
        language = kwargs.get('language', self.language)
        if language not in (None, '', 'auto', *LANGUAGES):
            raise ValueError(f'Unsupported Volcengine ASR language: {language}')
        url = _speech_base(self.api_base) + '/api/v3/auc/bigmodel/recognize/flash'
        headers = {
            'X-Api-Key': self.api_key,
            'X-Api-Resource-Id': RESOURCE_ID,
            'X-Api-Request-Id': str(uuid.uuid4()),
            'X-Api-Sequence': '-1',
            'Content-Type': 'application/json',
        }
        audio = {'format': fmt, 'data': base64.b64encode(path.read_bytes()).decode('ascii')}
        if fmt == 'ogg':
            audio['codec'] = 'opus'
        if language not in (None, '', 'auto'):
            audio['language'] = LANGUAGES[language]
        payload = {'user': {'uid': 'evalscope'}, 'audio': audio, 'request': {'model_name': 'bigmodel'}}
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        code = response.headers.get('X-Api-Status-Code')
        if response.status_code != 200 or code != '20000000':
            raise RuntimeError(f'Volcengine ASR error: HTTP {response.status_code}, code {code}, '
                               f'{response.headers.get("X-Api-Message", "")[:200]}')
        result = response.json()
        text = result.get('result', {}).get('text')
        if not isinstance(text, str):
            raise RuntimeError('Volcengine ASR response missing result.text')
        return {'text': text, 'raw': result}
