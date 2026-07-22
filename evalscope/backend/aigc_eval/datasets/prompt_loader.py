"""Prompt dataset loaders for AIGC evaluation."""
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_prompts(dataset_name: str, limit: int = 100, custom_path: Optional[str] = None) -> List[str]:
    """Load prompts from a dataset.

    Args:
        dataset_name: Name of the dataset (drawbench, coco_captions, parti, custom)
        limit: Maximum number of prompts to load

    Returns:
        List of text prompts
    """
    if dataset_name == 'drawbench':
        return _load_drawbench(limit)
    elif dataset_name == 'coco_captions':
        return _load_coco_captions(limit)
    elif dataset_name == 'parti':
        return _load_parti_prompts(limit)
    elif dataset_name == 'custom':
        return _load_custom_prompts(custom_path, limit)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')


def _load_drawbench(limit: int) -> List[str]:
    """Load DrawBench prompts."""
    # DrawBench: 200 prompts across 11 categories
    # Reference: https://arxiv.org/abs/2205.11487
    prompts = [
        'A red colored car',
        'A black colored car',
        'A pink colored car',
        'A white colored car',
        'A silver colored car',
        'A blue colored car',
        'A green colored car',
        'A yellow colored car',
        'A purple colored car',
        'An orange colored car',
        'A brown colored car',
        'A dog sitting on a chair',
        'A cat sitting on a chair',
        'A bird sitting on a chair',
        'A horse sitting on a chair',
        'A rabbit sitting on a chair',
        'A bear sitting on a chair',
        'A lion sitting on a chair',
        'A tiger sitting on a chair',
        'A monkey sitting on a chair',
        'A panda sitting on a chair',
        'A zebra sitting on a chair',
        'A photo of a cat',
        'A photo of a dog',
        'A photo of a horse',
        'A photo of a sheep',
        'A photo of a cow',
        'A photo of an elephant',
        'A photo of a bear',
        'A photo of a zebra',
        'A photo of a giraffe',
        'A photo of a hippopotamus',
        'A photo of a rhinoceros',
        'A photo of a kangaroo',
        'A painting of a cat',
        'A painting of a dog',
        'A painting of a horse',
        'A painting of a sheep',
        'A painting of a cow',
        'A painting of an elephant',
        'A painting of a bear',
        'A painting of a zebra',
        'A painting of a giraffe',
        'A painting of a hippopotamus',
        'A painting of a rhinoceros',
        'A painting of a kangaroo',
        'A sketch of a cat',
        'A sketch of a dog',
        'A sketch of a horse',
        'A sketch of a sheep',
        'A sketch of a cow',
        'A sketch of an elephant',
        'A sketch of a bear',
        'A sketch of a zebra',
        'A sketch of a giraffe',
        'A sketch of a hippopotamus',
        'A sketch of a rhinoceros',
        'A sketch of a kangaroo',
        'A cat and a dog',
        'A cat and a horse',
        'A cat and a sheep',
        'A cat and a cow',
        'A cat and an elephant',
        'A cat and a bear',
        'A cat and a zebra',
        'A cat and a giraffe',
        'A dog and a horse',
        'A dog and a sheep',
        'A dog and a cow',
        'A dog and an elephant',
        'A dog and a bear',
        'A dog and a zebra',
        'A dog and a giraffe',
        'A horse and a sheep',
        'A horse and a cow',
        'A horse and an elephant',
        'A horse and a bear',
        'A horse and a zebra',
        'A horse and a giraffe',
        'A sheep and a cow',
        'A sheep and an elephant',
        'A sheep and a bear',
        'A sheep and a zebra',
        'A sheep and a giraffe',
        'A cow and an elephant',
        'A cow and a bear',
        'A cow and a zebra',
        'A cow and a giraffe',
        'An elephant and a bear',
        'An elephant and a zebra',
        'An elephant and a giraffe',
        'A bear and a zebra',
        'A bear and a giraffe',
        'A zebra and a giraffe',
        'A cat with glasses',
        'A dog with glasses',
        'A horse with glasses',
        'A sheep with glasses',
        'A cow with glasses',
        'An elephant with glasses',
        'A bear with glasses',
        'A zebra with glasses',
        'A giraffe with glasses',
        'A hippopotamus with glasses',
        'A cat wearing a hat',
        'A dog wearing a hat',
        'A horse wearing a hat',
        'A sheep wearing a hat',
        'A cow wearing a hat',
        'An elephant wearing a hat',
        'A bear wearing a hat',
        'A zebra wearing a hat',
        'A giraffe wearing a hat',
        'A hippopotamus wearing a hat',
        'A cat in a car',
        'A dog in a car',
        'A horse in a car',
        'A sheep in a car',
        'A cow in a car',
        'An elephant in a car',
        'A bear in a car',
        'A zebra in a car',
        'A giraffe in a car',
        'A hippopotamus in a car',
        'A cat on a bicycle',
        'A dog on a bicycle',
        'A horse on a bicycle',
        'A sheep on a bicycle',
        'A cow on a bicycle',
        'An elephant on a bicycle',
        'A bear on a bicycle',
        'A zebra on a bicycle',
        'A giraffe on a bicycle',
        'A hippopotamus on a bicycle',
        'A cat playing piano',
        'A dog playing piano',
        'A horse playing piano',
        'A sheep playing piano',
        'A cow playing piano',
        'An elephant playing piano',
        'A bear playing piano',
        'A zebra playing piano',
        'A giraffe playing piano',
        'A hippopotamus playing piano',
        'A cat eating pizza',
        'A dog eating pizza',
        'A horse eating pizza',
        'A sheep eating pizza',
        'A cow eating pizza',
        'An elephant eating pizza',
        'A bear eating pizza',
        'A zebra eating pizza',
        'A giraffe eating pizza',
        'A hippopotamus eating pizza',
        'A cat reading a book',
        'A dog reading a book',
        'A horse reading a book',
        'A sheep reading a book',
        'A cow reading a book',
        'An elephant reading a book',
        'A bear reading a book',
        'A zebra reading a book',
        'A giraffe reading a book',
        'A hippopotamus reading a book',
        'A cat surfing',
        'A dog surfing',
        'A horse surfing',
        'A sheep surfing',
        'A cow surfing',
        'An elephant surfing',
        'A bear surfing',
        'A zebra surfing',
        'A giraffe surfing',
        'A hippopotamus surfing',
        'A cat in space',
        'A dog in space',
        'A horse in space',
        'A sheep in space',
        'A cow in space',
        'An elephant in space',
        'A bear in space',
        'A zebra in space',
        'A giraffe in space',
        'A hippopotamus in space',
        'A cat as a superhero',
        'A dog as a superhero',
        'A horse as a superhero',
        'A sheep as a superhero',
        'A cow as a superhero',
        'An elephant as a superhero',
        'A bear as a superhero',
        'A zebra as a superhero',
        'A giraffe as a superhero',
        'A hippopotamus as a superhero',
    ]
    return prompts[:limit]


def _load_coco_captions(limit: int) -> List[str]:
    """Load COCO Captions (placeholder)."""
    logger.warning('COCO Captions dataset not yet implemented, using default prompts')
    return _get_default_prompts(limit)


def _load_parti_prompts(limit: int) -> List[str]:
    """Load PartiPrompts from builtin JSON."""
    builtin_dir = Path(__file__).parent / 'builtin'
    json_path = builtin_dir / 'parti_prompts.json'
    if json_path.exists():
        with open(json_path) as f:
            prompts = json.load(f)
        logger.info(f'Loaded {len(prompts[:limit])} prompts from builtin PartiPrompts')
        return prompts[:limit]
    logger.warning('PartiPrompts JSON not found, using default prompts')
    return _get_default_prompts(limit)


def _get_default_prompts(limit: int) -> List[str]:
    """Get default prompts for testing."""
    default = [
        'A beautiful sunset over the ocean',
        'A majestic mountain landscape',
        'A bustling city street at night',
        'A peaceful forest with sunlight',
        'A colorful flower garden',
        'A snow-covered winter village',
        'A tropical beach with palm trees',
        'A medieval castle on a hill',
        'A futuristic city with flying cars',
        'A cozy cabin in the woods',
    ]
    return default[:limit]


def _load_custom_prompts(custom_path: Optional[str], limit: int) -> List[str]:
    """Load prompts from a custom file (one prompt per line)."""
    if not custom_path:
        logger.warning('Custom dataset selected but no path provided, using default prompts')
        return _get_default_prompts(limit)

    path = Path(custom_path)
    if not path.exists():
        logger.warning(f'Custom dataset file not found: {custom_path}, using default prompts')
        return _get_default_prompts(limit)

    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    logger.info(f'Loaded {len(lines[:limit])} prompts from custom file: {custom_path}')
    return lines[:limit]


def load_video_prompts(dataset_name: str, limit: int = 100, custom_path: Optional[str] = None) -> List[str]:
    """Load video generation prompts from a dataset.

    Args:
        dataset_name: Name of the dataset (msr_vtt, activitynet, custom)
        limit: Maximum number of prompts to load

    Returns:
        List of text prompts suitable for video generation
    """
    if dataset_name == 'msr_vtt':
        return _load_msr_vtt_prompts(limit)
    elif dataset_name == 'activitynet':
        return _load_activitynet_prompts(limit)
    elif dataset_name == 'custom':
        return _load_custom_prompts(custom_path, limit)
    else:
        raise ValueError(f'Unknown video dataset: {dataset_name}')


def _load_msr_vtt_prompts(limit: int) -> List[str]:
    """Load MSR-VTT prompts (placeholder — subset of video captions)."""
    prompts = [
        'A person is playing a musical instrument',
        'A crowd of people walking on a street',
        'A person is cooking food in a kitchen',
        'Cars driving on a highway',
        'A person is talking to the camera',
        'People dancing at a party',
        'A sports game is being played on a field',
        'A person is giving a presentation',
        'Someone is swimming in a pool',
        'An animal is running in a field',
        'A person is riding a bicycle',
        'People are sitting and eating at a restaurant',
        'A person is painting a picture',
        'A band is performing on stage',
        'Someone is typing on a computer',
        'A person is walking a dog in a park',
        'Birds flying in the sky at sunset',
        'A waterfall flowing down a mountain',
        'A firework display lighting up the night sky',
        'A train arriving at a station platform',
    ]
    return prompts[:limit]


def _load_activitynet_prompts(limit: int) -> List[str]:
    """Load ActivityNet Captions prompts (placeholder)."""
    logger.warning('ActivityNet Captions not yet implemented, using default video prompts')
    return _get_default_video_prompts(limit)


def _get_default_video_prompts(limit: int) -> List[str]:
    """Get default video prompts for testing."""
    default = [
        'A serene lake at sunset with gentle ripples',
        'A busy city intersection with cars and pedestrians',
        'A cat playing with a ball of yarn',
        'Waves crashing on a rocky shore',
        'A chef preparing a gourmet meal',
        'Aerial view of a winding mountain road',
        'A dog chasing butterflies in a meadow',
        'Rain falling on a city street at night',
        'A dancer performing a contemporary routine',
        'Time-lapse of clouds moving over mountains',
    ]
    return default[:limit]
