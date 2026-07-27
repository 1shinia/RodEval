"""TTS model adapter — local (edge-tts: Microsoft free TTS via HTTP).

Uses edge-tts library to call Microsoft Edge TTS API (free, no key needed).
Purely local Python process, no GPU or model download required.
"""
import asyncio
import logging
import tempfile
from typing import Any, Dict

from .base import AudioModelBase

logger = logging.getLogger(__name__)

# Voice mapping: frontend voice → edge-tts ShortName
_VOICE_MAP = {
    'xiaoxiao': 'zh-CN-XiaoxiaoNeural',
    'xiaoyi': 'zh-CN-XiaoyiNeural',
    'yunjian': 'zh-CN-YunjianNeural',
    'yunxi': 'zh-CN-YunxiNeural',
    'yunyang': 'zh-CN-YunyangNeural',
    'xiaobei': 'zh-CN-liaoning-XiaobeiNeural',
    'xiaoni': 'zh-CN-shaanxi-XiaoniNeural',
}
_DEFAULT_VOICE = 'zh-CN-XiaoxiaoNeural'


def _resolve_voice(voice: str) -> str:
    """Resolve voice name to edge-tts ShortName."""
    return _VOICE_MAP.get(voice.lower(), _DEFAULT_VOICE)


class TtsModelLocal(AudioModelBase):
    """Local TTS via edge-tts — no GPU, no model download, free."""

    def _generate_api(self, prompt: str, **kwargs) -> bytes:
        """Generate TTS audio via edge-tts."""
        voice = kwargs.get(
            'voice', self.config.get('voice', 'xiaoxiao')
        )
        speed = float(kwargs.get('speed', self.config.get('speed', 1.0)))
        short_name = _resolve_voice(voice)

        logger.info(
            f'Local TTS (edge-tts): voice={voice}→{short_name}, '
            f'speed={speed}'
        )

        async def _generate():
            import edge_tts

            rate = f'{int((speed - 1) * 100):+d}%' if speed != 1.0 else '+0%'
            tts = edge_tts.Communicate(prompt, short_name, rate=rate)
            with tempfile.NamedTemporaryFile(
                suffix='.mp3', delete=False
            ) as tmp:
                await tts.save(tmp.name)
                return tmp.name

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In a running loop, use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    _generate(), loop
                )
                tmp_path = future.result(timeout=60)
            else:
                tmp_path = asyncio.run(_generate())
        except RuntimeError:
            tmp_path = asyncio.run(_generate())

        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()

        import os
        os.unlink(tmp_path)

        logger.info(
            f'Local TTS generated {len(audio_bytes)} bytes'
        )
        return audio_bytes

    def _generate_local(self, prompt: str, **kwargs) -> bytes:
        return self._generate_api(prompt, **kwargs)

    def generate(self, prompt: str, **kwargs) -> bytes:
        return self._generate_api(prompt, **kwargs)
