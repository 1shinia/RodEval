"""Audio utility functions."""
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


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
