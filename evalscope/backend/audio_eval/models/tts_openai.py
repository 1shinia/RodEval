"""TTS model adapter — OpenAI-compatible /v1/audio/speech API."""
import logging
from typing import Any, Dict

import requests

from .base import AudioModelBase, resolve_api_url

logger = logging.getLogger(__name__)


class TtsModel(AudioModelBase):
    """TTS model via OpenAI-compatible /v1/audio/speech API."""

    def _generate_api(self, prompt: str, **kwargs) -> bytes:
        """Call TTS API, return raw audio bytes."""
        url = resolve_api_url(self.api_base, 'tts')
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        voice = kwargs.get('voice', self.config.get('voice', 'alloy'))
        fmt = kwargs.get('response_format', self.config.get('response_format', 'mp3'))
        speed = float(kwargs.get('speed', self.config.get('speed', 1.0)))

        payload = {
            'model': self.model_name,
            'input': prompt,
            'voice': voice,
            'response_format': fmt,
            'speed': speed,
        }

        logger.info(
            f'TTS API request: url={url}, model={self.model_name}, '
            f'voice={voice}, fmt={fmt}, speed={speed}'
        )
        response = requests.post(url, json=payload, headers=headers, timeout=120)

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(f'TTS API error {response.status_code}: {error_text}')
            raise RuntimeError(f'TTS API returned {response.status_code}: {error_text}')

        audio_bytes = response.content
        logger.info(f'TTS generated {len(audio_bytes)} bytes')
        return audio_bytes

    def _generate_local(self, prompt: str, **kwargs) -> bytes:
        raise NotImplementedError('Local TTS model loading not yet implemented')

    def generate(self, prompt: str, **kwargs) -> bytes:
        """Generate speech from text. Returns raw audio bytes."""
        if self.api_base and self.api_key:
            return self._generate_api(prompt, **kwargs)
        else:
            return self._generate_local(prompt, **kwargs)
