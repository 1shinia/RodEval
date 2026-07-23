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
    elif dataset_name == 'custom':
        return _load_custom_prompts(custom_path, limit)
    else:
        raise ValueError(f'Unknown video dataset: {dataset_name}')


def _load_msr_vtt_prompts(limit: int) -> List[str]:
    """Load built-in video prompts — diverse scenes for text-to-video evaluation.

    Covers 8 categories: action/sports, nature/landscape, urban/city,
    animals, people/daily, weather/phenomena, vehicles/transport, art/performance.
    """
    prompts = [
        # --- Action & Sports ---
        'A soccer player scoring a goal with a powerful kick',
        'A basketball player making a slam dunk',
        'A surfer riding a large wave at sunset',
        'A skateboarder performing a kickflip on a ramp',
        'A gymnast performing a floor routine',
        'A boxer throwing rapid punches at a heavy bag',
        'A rock climber scaling a steep cliff face',
        'A marathon runner crossing the finish line',
        'A swimmer diving into a pool from a starting block',
        'A tennis player serving the ball with full force',
        # --- Nature & Landscape ---
        'A waterfall cascading down a rocky cliff in a lush forest',
        'Waves crashing against rocks on a rugged coastline',
        'A time-lapse of clouds rolling over mountain peaks',
        'A river winding through a green valley at golden hour',
        'A volcanic eruption with lava flowing down the slope',
        'A field of sunflowers swaying in the wind',
        'Heavy rain falling on a dense tropical rainforest',
        'The northern lights shimmering across a snowy sky',
        'A thunderstorm with lightning striking the horizon',
        'Snow falling gently on a quiet pine forest',
        # --- Urban & City ---
        'A busy intersection with cars and pedestrians crossing',
        'A subway train arriving at a crowded station platform',
        'An aerial view of a city skyline at night with traffic trails',
        'A street market with vendors and colorful stalls',
        'People walking through a modern glass shopping mall',
        'Construction workers on scaffolding building a skyscraper',
        'A coffee shop barista preparing a latte with latte art',
        'Raindrops on a window overlooking a neon-lit city street',
        'A food truck serving customers at a busy night market',
        'A cyclist weaving through city traffic during rush hour',
        # --- Animals ---
        'A cheetah sprinting across the African savanna',
        'Dolphins leaping out of the ocean in perfect arcs',
        'A cat stalking and pouncing on a toy mouse',
        'Birds migrating in a V-formation across a sunset sky',
        'A dog catching a frisbee mid-air in a park',
        'A school of colorful fish swimming through a coral reef',
        'A horse galloping across an open meadow',
        'A hummingbird hovering and drinking nectar from a flower',
        'A bear catching salmon in a rushing river',
        'A butterfly emerging from its chrysalis in slow motion',
        # --- People & Daily Life ---
        'A chef tossing vegetables in a flaming wok',
        'A musician playing a grand piano on a concert stage',
        'A painter creating a landscape on a canvas with oil paints',
        'Children blowing out candles on a birthday cake',
        'A person reading a book by a fireplace on a rainy evening',
        'A couple dancing a waltz in a grand ballroom',
        'A barber cutting a customer hair with scissors',
        'A gardener planting flowers in a backyard garden',
        'A photographer taking pictures at a wedding ceremony',
        'An elderly couple walking hand in hand along a beach',
        # --- Weather & Natural Phenomena ---
        'A tornado forming and touching down over open farmland',
        'A sandstorm rolling across a vast desert landscape',
        'Fireworks exploding in a cascade of colors over a city',
        'A solar eclipse darkening the sky with a glowing corona',
        'Leaves falling and swirling in an autumn breeze',
        'A river of lava slowly advancing down a volcanic slope',
        'Snowflakes drifting down on a Christmas market',
        'A rainbow forming after a heavy summer rainstorm',
        'Fog rolling over a suspension bridge at dawn',
        'A meteor shower streaking across a star-filled night sky',
        # --- Vehicles & Transport ---
        'A high-speed train passing through countryside at sunset',
        'A rocket launching from a launchpad with massive exhaust',
        'A hot air balloon rising over patchwork farmland',
        'A sailboat gliding across calm blue waters at sunrise',
        'A fighter jet performing an aerial loop at an airshow',
        'A vintage car driving along a winding coastal road',
        'A cargo ship navigating through icy arctic waters',
        'A helicopter landing on a rooftop helipad in a city',
        'A cable car climbing up a steep mountain slope',
        'A motorcycle racing on a circuit track at high speed',
        # --- Art & Performance ---
        'A ballet dancer performing a pirouette on stage',
        'A street performer juggling flaming torches at night',
        'A calligrapher writing Chinese characters with a brush',
        'A sculptor chiseling a marble statue in a studio',
        'A marching band parading down a decorated street',
        'A theater actor delivering a dramatic monologue',
        'A puppet show being performed for a crowd of children',
        'A DJ mixing tracks at a festival with colorful lasers',
        'A magician pulling a rabbit out of a top hat',
        'A traditional lion dance performance during a festival',
    ]
    return prompts[:limit]

