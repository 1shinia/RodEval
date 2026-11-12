"""Audio utility functions."""
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


def validate_tts_audio(data: bytes, fmt: str) -> float:
    """Reject corrupt or mislabeled TTS bytes before storing a successful sample."""
    from io import BytesIO

    if not isinstance(data, bytes) or not data:
        raise ValueError('TTS returned empty or invalid audio')
    headers = {
        'wav': data.startswith(b'RIFF') and data[8:12] == b'WAVE',
        'mp3': data.startswith(b'ID3') or (len(data) >= 2 and data[0] == 255 and data[1] & 224 == 224),
        'ogg': data.startswith(b'OggS'),
    }
    if not headers.get(fmt.lower(), False):
        raise ValueError(f'TTS returned invalid {fmt} audio')
    try:
        from mutagen.mp3 import MP3
        from mutagen.wave import WAVE
        from mutagen.oggopus import OggOpus
        from mutagen.oggvorbis import OggVorbis
        if fmt.lower() == 'wav':
            import wave
            with wave.open(BytesIO(data), 'rb') as wav:
                expected = wav.getnframes() * wav.getnchannels() * wav.getsampwidth()
                if len(wav.readframes(wav.getnframes())) != expected:
                    raise ValueError('Truncated WAV data')
        ogg_parser = OggOpus if b'OpusHead' in data[:128] else OggVorbis
        parser = {'mp3': MP3, 'wav': WAVE, 'ogg': ogg_parser}[fmt.lower()]
        audio = parser(BytesIO(data))
        duration = float(audio.info.length)
    except Exception as exc:
        raise ValueError(f'TTS returned undecodable {fmt} audio') from exc
    if duration <= 0:
        raise ValueError(f'TTS returned undecodable {fmt} audio')
    return round(duration, 2)


def save_audio(data: bytes, output_path: Union[str, Path], fmt: str = 'mp3') -> Path:
    """Save raw audio bytes to file."""
    path = Path(output_path)
    if path.suffix.lower() != f'.{fmt}':
        path = path.with_suffix(f'.{fmt}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    logger.info(f'Audio saved: {path} ({len(data)} bytes)')
    return path


def get_audio_duration(file_path: Union[str, Path]) -> Optional[float]:
    """Get audio duration in seconds. Returns None if unreadable."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(str(file_path))
        return round(audio.info.length, 2)
    except Exception:
        pass
    try:
        from mutagen.wave import WAVE
        audio = WAVE(str(file_path))
        return round(audio.info.length, 2)
    except Exception:
        pass
    return None
