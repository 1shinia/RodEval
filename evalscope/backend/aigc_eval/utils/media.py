"""Media file utilities for AIGC evaluation."""
import logging
from pathlib import Path
from PIL import Image
from typing import List

logger = logging.getLogger(__name__)


def save_images(images: List[Image.Image], output_dir: Path, prefix: str = '') -> List[Path]:
    """Save images to output directory.

    Args:
        images: List of PIL Images
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, img in enumerate(images):
        filename = f'{prefix}{i:04d}.png' if prefix else f'{i:04d}.png'
        path = output_dir / filename
        img.save(path)
        paths.append(path)

    logger.info(f'Saved {len(paths)} images to {output_dir}')
    return paths


def create_thumbnails(
    images: List[Image.Image],
    output_paths: List[Path],
    size: tuple = (256, 256),
) -> None:
    """Create thumbnails for images.

    Args:
        images: List of PIL Images
        output_paths: List of output paths for thumbnails
        size: Thumbnail size (width, height)
    """
    if len(images) != len(output_paths):
        raise ValueError('Number of images must match number of output paths')

    for img, path in zip(images, output_paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        thumb = img.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        thumb.save(path, quality=85)

    logger.info(f'Created {len(output_paths)} thumbnails')
