"""TTS model adapter — DashScope (阿里云百炼) native API (HTTP)."""
import logging
from typing import Any, Dict

import requests

from .asr_dashscope import _base_url
from .base import AudioModelBase

logger = logging.getLogger(__name__)


class TtsModelDashScope(AudioModelBase):
    """TTS via DashScope HTTP API — /services/audio/tts/SpeechSynthesizer."""

    @property
    def _tts_url(self) -> str:
        base = _base_url(self.api_base)
        return f'{base}/api/v1/services/audio/tts/SpeechSynthesizer'

    def _generate_api(self, prompt: str, **kwargs) -> bytes:
        url = self._tts_url
        voice = kwargs.get('voice', self.config.get('voice', 'longxiaochun'))
        fmt = kwargs.get('response_format', self.config.get('response_format', 'mp3'))
        speed = float(kwargs.get('speed', self.config.get('speed', 1.0)))

        payload = {
            'model': self.model_name,
            'input': {
                'text': prompt,
                'voice': voice,
                'format': fmt,
            },
        }
        # Add optional speed control
        if speed != 1.0:
            payload['input']['rate'] = speed

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        logger.info(
            f'DashScope TTS: url={url}, model={self.model_name}, '
            f'voice={voice}, fmt={fmt}, speed={speed}'
        )
        response = requests.post(url, json=payload, headers=headers, timeout=120)

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f'DashScope TTS error {response.status_code}: {error_text}')
            raise RuntimeError(f'DashScope TTS returned {response.status_code}: {error_text}')

        data = response.json()
        # Response: {"output": {"audio": {"url": "...", "id": "..."}}}
        audio_info = data.get('output', {}).get('audio', {})
        audio_url = audio_info.get('url', '')

        if not audio_url:
            # Fallback: try direct base64 audio in output
            audio_b64 = data.get('output', {}).get('audio', {}).get('data') or data.get('output', {}).get('audio')
            if isinstance(audio_b64, str) and audio_b64:
                import base64
                audio_bytes = base64.b64decode(audio_b64)
                logger.info(f'DashScope TTS generated {len(audio_bytes)} bytes (base64)')
                return audio_bytes
            raise RuntimeError(f'DashScope TTS: no audio in response: {str(data)[:200]}')

        # Download audio from URL
        logger.info(f'DashScope TTS downloading audio from: {audio_url[:80]}...')
        audio_resp = requests.get(audio_url, timeout=60)
        if audio_resp.status_code != 200:
            raise RuntimeError(f'Failed to download TTS audio: {audio_resp.status_code}')

        audio_bytes = audio_resp.content
        logger.info(f'DashScope TTS generated {len(audio_bytes)} bytes')
        return audio_bytes

    def _generate_local(self, prompt: str, **kwargs) -> bytes:
        raise NotImplementedError('Local TTS model loading not yet implemented')

    def generate(self, prompt: str, **kwargs) -> bytes:
        return self._generate_api(prompt, **kwargs)
