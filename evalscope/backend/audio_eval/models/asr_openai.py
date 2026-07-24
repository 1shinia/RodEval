"""ASR model adapter — OpenAI Whisper-compatible API."""
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .base import AudioModelBase, resolve_api_url

logger = logging.getLogger(__name__)


class AsrModel(AudioModelBase):
    """ASR model via OpenAI-compatible /v1/audio/transcriptions API."""

    def _generate_api(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Call ASR API with audio file upload.

        Uses multipart/form-data per OpenAI Whisper API spec.
        """
        url = resolve_api_url(self.api_base, 'asr')
        headers = {'Authorization': f'Bearer {self.api_key}'}

        language = kwargs.get('language', self.language)
        if language == 'auto':
            language = None

        data = {'model': self.model_name}
        if language:
            data['language'] = language

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f'Audio file not found: {audio_path}')

        mime_type = _guess_mime(audio_path_obj.suffix)
        with open(audio_path_obj, 'rb') as f:
            files = {'file': (audio_path_obj.name, f, mime_type)}

            logger.info(f'ASR API request: url={url}, model={self.model_name}, '
                        f'file={audio_path_obj.name}, language={language}')
            response = requests.post(
                url, headers=headers, data=data, files=files, timeout=120
            )

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f'ASR API error {response.status_code}: {error_text}')
            raise RuntimeError(f'ASR API returned {response.status_code}: {error_text}')

        result = response.json()
        transcription = result.get('text', '')
        logger.info(f'ASR result: {transcription[:100]}...')
        return {'text': transcription, 'raw': result}

    def _generate_local(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Local ASR — placeholder, not implemented yet."""
        raise NotImplementedError('Local ASR model loading not yet implemented')

    def generate(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Run ASR on audio file. Returns dict with 'text' key."""
        if self.api_base and self.api_key:
            return self._generate_api(audio_path, **kwargs)
        else:
            return self._generate_local(audio_path, **kwargs)


def _guess_mime(suffix: str) -> str:
    """Guess MIME type from file extension."""
    mapping = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.webm': 'audio/webm',
    }
    return mapping.get(suffix.lower(), 'audio/wav')
