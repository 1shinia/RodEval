"""Base class for audio models + API URL resolution."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioModelBase:
    """Base class for audio models (ASR/TTS)."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_name_or_path', '')
        self.api_base = config.get('api_base', '')
        self.api_key = config.get('api_key', '')
        self.language = config.get('language', 'zh')
        self._model = None

    def load(self):
        """Load model (no-op for API mode)."""
        pass

    def unload(self):
        """Unload model."""
        self._model = None

    def generate(self, *args, **kwargs):
        raise NotImplementedError


def resolve_api_url(api_base: str, tool: str) -> str:
    """Resolve the full API endpoint URL from a base URL + tool type.

    ASR: {base}/audio/transcriptions
    TTS: {base}/audio/speech
    """
    if not api_base:
        return api_base

    url = api_base.rstrip('/')
    lower = url.lower()

    # If URL already contains a specific endpoint path, use as-is
    if any(kw in lower for kw in ('/audio/', '/transcriptions', '/speech', '/synthesize')):
        return url

    if tool == 'asr':
        return f'{url}/audio/transcriptions'
    elif tool == 'tts':
        return f'{url}/audio/speech'
    return url
