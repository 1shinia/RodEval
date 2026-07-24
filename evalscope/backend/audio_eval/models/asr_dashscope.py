"""ASR model adapter — DashScope (阿里云百炼) native API.

Uses DashScope file upload to convert local audio to a file_url,
then calls the ASR transcription API.
"""
import json
import logging
from typing import Any, Dict

import requests

from .base import AudioModelBase

logger = logging.getLogger(__name__)

# Standard DashScope endpoints (fallback)
_DEFAULT_BASE = 'https://dashscope.aliyuncs.com'


def _base_url(api_base: str) -> str:
    """Derive the DashScope base URL from api_base, stripping /api/v1 suffix."""
    if not api_base:
        return _DEFAULT_BASE
    url = api_base.rstrip('/')
    if url.endswith('/api/v1'):
        url = url[:-len('/api/v1')]
    elif url.endswith('/compatible-mode/v1'):
        url = url[:-len('/compatible-mode/v1')]
    return url if url.startswith('http') else _DEFAULT_BASE


class AsrModelDashScope(AudioModelBase):
    """ASR via DashScope — uploads local file first, then transcribes via file_urls."""

    @property
    def _files_url(self) -> str:
        return f'{_base_url(self.api_base)}/api/v1/files'

    @property
    def _asr_url(self) -> str:
        return f'{_base_url(self.api_base)}/api/v1/services/audio/asr/transcription'

    def _upload_file(self, audio_path: str) -> str:
        """Upload audio file to DashScope, return accessible URL."""
        from pathlib import Path

        p = Path(audio_path)
        mime = _guess_mime(p.suffix)
        base = _base_url(self.api_base)

        with open(p, 'rb') as f:
            resp = requests.post(
                self._files_url,
                headers={'Authorization': f'Bearer {self.api_key}'},
                files={'file': (p.name, f, mime)},
                timeout=120,
            )

        if resp.status_code != 200:
            error_text = resp.text[:500]
            logger.error(f'DashScope file upload error {resp.status_code}: {error_text}')
            raise RuntimeError(f'DashScope file upload failed: {resp.status_code} {error_text}')

        data = resp.json()
        data_wrapper = data.get('data', data)
        uploaded = data_wrapper.get('uploaded_files', [])
        file_id = ''
        if uploaded:
            file_id = uploaded[0].get('file_id', '')
        if not file_id:
            raise RuntimeError(f'DashScope file upload: no file_id in response: {json.dumps(data)[:200]}')

        file_url = f'{base}/api/v1/files/{file_id}/content'
        logger.info(f'DashScope file uploaded: {file_id} → {file_url}')
        return file_url

    def _generate_api(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Upload file → get URL → call ASR transcription."""
        language = kwargs.get('language', self.language)

        file_url = self._upload_file(audio_path)

        payload = {
            'model': self.model_name,
            'input': {
                'file_urls': [file_url],
            },
            'parameters': {},
        }
        if language and language != 'auto':
            payload['parameters']['language'] = language

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        logger.info(
            f'DashScope ASR: url={self._asr_url}, model={self.model_name}, '
            f'lang={language}'
        )
        response = requests.post(self._asr_url, json=payload, headers=headers, timeout=300)

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f'DashScope ASR error {response.status_code}: {error_text}')
            raise RuntimeError(f'DashScope ASR returned {response.status_code}: {error_text}')

        data = response.json()
        output = data.get('output', {})
        transcription = output.get('text', '')
        logger.info(f'DashScope ASR result: {transcription[:100]}...')
        return {'text': transcription, 'raw': data}

    def _generate_local(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError('Local ASR model loading not yet implemented')

    def generate(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        return self._generate_api(audio_path, **kwargs)


def _guess_mime(suffix: str) -> str:
    mapping = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.webm': 'audio/webm',
    }
    return mapping.get(suffix.lower(), 'audio/wav')
