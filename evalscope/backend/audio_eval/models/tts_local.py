"""TTS model adapter — local (edge-tts: Microsoft free TTS via HTTP).

Uses edge-tts library to call Microsoft Edge TTS API (free, no key needed).
Purely local Python process, no GPU or model download required.
"""
import asyncio
import logging
import re
import tempfile
from typing import Any, Dict, Optional

from .base import AudioModelBase

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

# Voice alias mapping: short alias → edge-tts ShortName
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

# Full edge-tts ShortName pattern, e.g. zh-CN-XiaoxiaoNeural, en-US-AriaNeural
_SHORTNAME_RE = re.compile(r'^[a-z]{2,3}-[A-Z]{2,}-[A-Za-z0-9]+Neural$')


def _resolve_voice(voice: Optional[str]) -> str:
    """Resolve voice name to edge-tts ShortName.

    Accepts either a short alias from _VOICE_MAP or a full edge-tts
    ShortName (e.g. 'zh-CN-XiaoxiaoNeural'). Empty input returns the
    default; an unknown alias raises instead of silently falling back
    so users notice misconfigured voices instead of hearing the wrong
    voice in the report.
    """
    v = (voice or '').strip()
    if not v:
        return _DEFAULT_VOICE
    if _SHORTNAME_RE.match(v):
        return v
    mapped = _VOICE_MAP.get(v.lower())
    if mapped is None:
        raise ValueError(
            f'edge-tts 不支持音色 {voice!r}。'
            f'可选别名: {sorted(_VOICE_MAP.keys())}; '
            f'或直接填完整 ShortName (如 zh-CN-XiaoxiaoNeural)'
        )
    return mapped


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
