"""Audio utility functions."""
import logging
from pathlib import Path
from typing import Union

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
