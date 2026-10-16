"""Volcengine native TTS 2.0 HTTP streaming adapter."""
import base64
import binascii
import json
import uuid
from typing import Any

import requests

from .asr_volcengine import _speech_base
from .base import AudioModelBase


class TtsModelVolcengine(AudioModelBase):
    """Collect Base64 audio chunks from the native unidirectional HTTP API."""

    def generate(self, prompt: str, **kwargs) -> bytes:
        resource = self.model_name
        if resource not in ('seed-tts-2.0', 'seed-icl-2.0'):
            raise ValueError('Volcengine TTS model must be seed-tts-2.0 or seed-icl-2.0')
        voice = kwargs.get('voice', self.config.get('voice', ''))
        if not isinstance(voice, str) or not voice.strip() or voice.strip() == 'alloy':
            raise ValueError('Volcengine TTS voice must be a valid speaker ID from the voice console')
        fmt = kwargs.get('response_format', 'mp3')
        if fmt == 'ogg':
            fmt = 'ogg_opus'
        if fmt not in ('mp3', 'wav', 'ogg_opus'):
            raise ValueError('Volcengine TTS format must be mp3, wav or ogg_opus')
        speed = float(kwargs.get('speed', 1.0))
        if not 0.5 <= speed <= 2.0:
            raise ValueError('Volcengine TTS speed must be between 0.5 and 2.0')
        audio_params: dict[str, Any] = {'format': fmt}
        if speed != 1.0:
            audio_params['speech_rate'] = round((speed - 1) * 100)
        url = _speech_base(self.api_base) + '/api/v3/tts/unidirectional'
        headers = {
            'X-Api-Key': self.api_key,
            'X-Api-Resource-Id': resource,
            'X-Api-Request-Id': str(uuid.uuid4()),
            'Content-Type': 'application/json',
        }
        request_params = {
            'text': prompt, 'speaker': voice.strip(), 'audio_params': audio_params,
        }
        # The cloned-voice resource requires an explicit synthesis model.
        if resource == 'seed-icl-2.0':
            request_params['model'] = kwargs.get('tts_model', 'seed-tts-2.0-standard')
        response = requests.post(url, headers=headers, json={'req_params': request_params}, stream=True, timeout=120)
        try:
            if response.status_code != 200:
                raise RuntimeError(f'Volcengine TTS HTTP {response.status_code}')
            audio = bytearray()
            for line in response.iter_lines():
                if not line:
                    continue
                packet = json.loads(line)
                code = packet.get('code')
                if code == 20000000:
                    break
                if code != 0:
                    raise RuntimeError(f'Volcengine TTS error {code}: {packet.get("message", "")[:200]}')
                if packet.get('data'):
                    try:
                        audio.extend(base64.b64decode(packet['data'], validate=True))
                    except (ValueError, binascii.Error) as exc:
                        raise RuntimeError('Volcengine TTS invalid audio chunk') from exc
            if not audio:
                raise RuntimeError('Volcengine TTS returned no audio')
            return bytes(audio)
        finally:
            response.close()
