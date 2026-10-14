"""Prompt dataset loaders for AIGC evaluation."""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))


def load_prompts(
    dataset_name: str,
    limit: int = 100,
    custom_path: Optional[str] = None,
    shuffle: bool = False,
    seed: int = 42
) -> List[str]:
    """Load prompts from a dataset.

    Args:
        dataset_name: Name of the dataset (drawbench, coco_captions, parti, custom)
        limit: Maximum number of prompts to load
        shuffle: Whether to randomly shuffle before taking limit
        seed: Random seed for shuffle reproducibility

    Returns:
        List of text prompts
    """
    # Load all available prompts, then optionally shuffle, then slice.
    # Always load everything so shuffle has a meaningful population.
    if dataset_name == 'drawbench':
        prompts = _load_drawbench(None)  # type: ignore[arg-type]
    elif dataset_name == 'coco_captions':
        prompts = _load_coco_captions(None)
    elif dataset_name == 'parti':
        prompts = _load_parti_prompts(None)
    elif dataset_name == 'custom':
        prompts = _load_custom_prompts(custom_path, None)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    if shuffle and len(prompts) > 1:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    return prompts[:limit]


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


def load_video_prompts(
    dataset_name: str,
    limit: int = 100,
    custom_path: Optional[str] = None,
    shuffle: bool = False,
    seed: int = 42
) -> List[str]:
    """Load video generation prompts from a dataset.

    Args:
        dataset_name: Name of the dataset (msr_vtt, msr_vtt_safe, custom)
        limit: Maximum number of prompts to load
        shuffle: Whether to randomly shuffle before taking limit
        seed: Random seed for shuffle reproducibility

    Returns:
        List of text prompts suitable for video generation
    """
    # Load all available prompts, then optionally shuffle, then slice.
    # Always load everything so shuffle has a meaningful population.
    if dataset_name in _VIDEO_DATASET_ORDERS:
        prompts = _builtin_video_prompts(_VIDEO_DATASET_ORDERS[dataset_name],
                                         degrade=dataset_name != 'msr_vtt')
    elif dataset_name == 'custom':
        prompts = _load_custom_prompts(custom_path, None)
    else:
        raise ValueError(f'Unknown video dataset: {dataset_name}')

    if shuffle and len(prompts) > 1:
        rng = random.Random(seed)
        rng.shuffle(prompts)

    return prompts[:limit]


def _builtin_video_prompts(category_order: tuple, degrade: bool = False) -> List[str]:
    """内置视频提示词的唯一数据源投影。

    ``degrade=True`` 用同类别、同句式的安全等价文本替换掉版权/危险/未成年语义的条目
    （风险分级见 references/msr-vtt-prompts.md）：条数、类别、句式一律不变，
    msr_vtt 的历史基线因此不受影响，同时给出一条不会被供应商风控挡住的默认集合。
    """
    by_category: Dict[str, List[str]] = {}
    for category, prompt, safe_prompt in _VIDEO_PROMPT_TABLE:
        by_category.setdefault(category, []).append(safe_prompt if degrade and safe_prompt else prompt)
    return [prompt for category in category_order for prompt in by_category[category]]


# (category, prompt, safe_prompt)：safe_prompt=None 表示该条本身无需降级。
# 依据：赛事转播、演出、军事、灾难、未成年人这几类语义在方舟/百炼的拒绝率明显更高，
# 实测 A soccer player scoring a goal with a powerful kick 被拒（copyright restrictions）。
_VIDEO_PROMPT_TABLE: List[tuple] = [
    # --- Action & Sports ---
    ('Action & Sports', 'A soccer player scoring a goal with a powerful kick', 'A soccer ball curving into a goal net on an empty field'),
    ('Action & Sports', 'A basketball player making a slam dunk', 'A basketball bouncing across an empty outdoor court'),
    ('Action & Sports', 'A surfer riding a large wave at sunset', 'A surfer gliding on gentle waves at sunrise'),
    ('Action & Sports', 'A skateboarder performing a kickflip on a ramp', 'A skateboarder rolling smoothly down a paved path'),
    ('Action & Sports', 'A gymnast performing a floor routine', None),
    ('Action & Sports', 'A boxer throwing rapid punches at a heavy bag', 'A dancer practicing quick footwork in a mirrored studio'),
    ('Action & Sports', 'A rock climber scaling a steep cliff face', 'A hiker walking along a mountain trail'),
    ('Action & Sports', 'A marathon runner crossing the finish line', 'Joggers running along a tree-lined park path at sunrise'),
    ('Action & Sports', 'A swimmer diving into a pool from a starting block', 'A swimmer gliding through clear blue water in a pool'),
    ('Action & Sports', 'A tennis player serving the ball with full force', 'A tennis ball leaving a puff of dust on a clay court'),
    # --- Nature & Landscape ---
    ('Nature & Landscape', 'A waterfall cascading down a rocky cliff in a lush forest', None),
    ('Nature & Landscape', 'Waves crashing against rocks on a rugged coastline', None),
    ('Nature & Landscape', 'A time-lapse of clouds rolling over mountain peaks', None),
    ('Nature & Landscape', 'A river winding through a green valley at golden hour', None),
    ('Nature & Landscape', 'A volcanic eruption with lava flowing down the slope', 'Steam rising from geothermal hot springs at dawn'),
    ('Nature & Landscape', 'A field of sunflowers swaying in the wind', None),
    ('Nature & Landscape', 'Heavy rain falling on a dense tropical rainforest', None),
    ('Nature & Landscape', 'The northern lights shimmering across a snowy sky', None),
    ('Nature & Landscape', 'A thunderstorm with lightning striking the horizon', 'Rain clouds gathering over a quiet green valley'),
    ('Nature & Landscape', 'Snow falling gently on a quiet pine forest', None),
    # --- Urban & City ---
    ('Urban & City', 'A busy intersection with cars and pedestrians crossing', None),
    ('Urban & City', 'A subway train arriving at a crowded station platform', None),
    ('Urban & City', 'An aerial view of a city skyline at night with traffic trails', None),
    ('Urban & City', 'A street market with vendors and colorful stalls', None),
    ('Urban & City', 'People walking through a modern glass shopping mall', None),
    ('Urban & City', 'Construction workers on scaffolding building a skyscraper', None),
    ('Urban & City', 'A coffee shop barista preparing a latte with latte art', None),
    ('Urban & City', 'Raindrops on a window overlooking a neon-lit city street', None),
    ('Urban & City', 'A food truck serving customers at a busy night market', None),
    ('Urban & City', 'A cyclist weaving through city traffic during rush hour', 'A tram passing a quiet morning street'),
    # --- Animals ---
    ('Animals', 'A cheetah sprinting across the African savanna', None),
    ('Animals', 'Dolphins leaping out of the ocean in perfect arcs', None),
    ('Animals', 'A cat stalking and pouncing on a toy mouse', None),
    ('Animals', 'Birds migrating in a V-formation across a sunset sky', None),
    ('Animals', 'A dog catching a frisbee mid-air in a park', None),
    ('Animals', 'A school of colorful fish swimming through a coral reef', None),
    ('Animals', 'A horse galloping across an open meadow', None),
    ('Animals', 'A hummingbird hovering and drinking nectar from a flower', None),
    ('Animals', 'A bear catching salmon in a rushing river', None),
    ('Animals', 'A butterfly emerging from its chrysalis in slow motion', None),
    # --- People & Daily Life ---
    ('People & Daily Life', 'A chef tossing vegetables in a flaming wok', None),
    ('People & Daily Life', 'A musician playing a grand piano on a concert stage', 'A musician playing a guitar in a cozy living room'),
    ('People & Daily Life', 'A painter creating a landscape on a canvas with oil paints', None),
    ('People & Daily Life', 'Children blowing out candles on a birthday cake', 'A birthday cake with candles flickering on a table'),
    ('People & Daily Life', 'A person reading a book by a fireplace on a rainy evening', None),
    ('People & Daily Life', 'A couple dancing a waltz in a grand ballroom', None),
    ('People & Daily Life', 'A barber cutting a customer hair with scissors', 'A barber styling a customer hair in a bright salon'),
    ('People & Daily Life', 'A gardener planting flowers in a backyard garden', None),
    ('People & Daily Life', 'A photographer taking pictures at a wedding ceremony', 'A photographer adjusting a camera on a tripod'),
    ('People & Daily Life', 'An elderly couple walking hand in hand along a beach', None),
    # --- Weather & Natural Phenomena ---
    ('Weather & Natural Phenomena', 'A tornado forming and touching down over open farmland', 'Tall grass swaying in a strong prairie wind'),
    ('Weather & Natural Phenomena', 'A sandstorm rolling across a vast desert landscape', 'Dust drifting in golden light over rolling dunes'),
    ('Weather & Natural Phenomena', 'Fireworks exploding in a cascade of colors over a city', None),
    ('Weather & Natural Phenomena', 'A solar eclipse darkening the sky with a glowing corona', None),
    ('Weather & Natural Phenomena', 'Leaves falling and swirling in an autumn breeze', None),
    ('Weather & Natural Phenomena', 'A river of lava slowly advancing down a volcanic slope', 'Sunlight reflecting off a dark rocky lava field'),
    ('Weather & Natural Phenomena', 'Snowflakes drifting down on a Christmas market', 'Snowflakes drifting down on a quiet village street'),
    ('Weather & Natural Phenomena', 'A rainbow forming after a heavy summer rainstorm', None),
    ('Weather & Natural Phenomena', 'Fog rolling over a suspension bridge at dawn', None),
    ('Weather & Natural Phenomena', 'A meteor shower streaking across a star-filled night sky', None),
    # --- Vehicles & Transport ---
    ('Vehicles & Transport', 'A high-speed train passing through countryside at sunset', None),
    ('Vehicles & Transport', 'A rocket launching from a launchpad with massive exhaust', 'A weather balloon rising into a clear blue sky'),
    ('Vehicles & Transport', 'A hot air balloon rising over patchwork farmland', None),
    ('Vehicles & Transport', 'A sailboat gliding across calm blue waters at sunrise', None),
    ('Vehicles & Transport', 'A fighter jet performing an aerial loop at an airshow', 'A glider plane soaring above green hills'),
    ('Vehicles & Transport', 'A vintage car driving along a winding coastal road', None),
    ('Vehicles & Transport', 'A cargo ship navigating through icy arctic waters', None),
    ('Vehicles & Transport', 'A helicopter landing on a rooftop helipad in a city', None),
    ('Vehicles & Transport', 'A cable car climbing up a steep mountain slope', None),
    ('Vehicles & Transport', 'A motorcycle racing on a circuit track at high speed', 'A motorcycle cruising along a countryside road'),
    # --- Art & Performance ---
    ('Art & Performance', 'A ballet dancer performing a pirouette on stage', None),
    ('Art & Performance', 'A street performer juggling flaming torches at night', 'A street performer juggling colorful balls'),
    ('Art & Performance', 'A calligrapher writing Chinese characters with a brush', None),
    ('Art & Performance', 'A sculptor chiseling a marble statue in a studio', None),
    ('Art & Performance', 'A marching band parading down a decorated street', 'A group of drummers practicing in a studio'),
    ('Art & Performance', 'A theater actor delivering a dramatic monologue', 'An actor rehearsing lines in an empty rehearsal room'),
    ('Art & Performance', 'A puppet show being performed for a crowd of children', 'A puppet show on a small wooden stage'),
    ('Art & Performance', 'A DJ mixing tracks at a festival with colorful lasers', 'A DJ mixing music in a small studio'),
    ('Art & Performance', 'A magician pulling a rabbit out of a top hat', None),
    ('Art & Performance', 'A traditional lion dance performance during a festival', None),
]


# 基线顺序：与历史 msr_vtt 完全一致（改这里等于改基线，历史分数不可比）
_DEFAULT_VIDEO_CATEGORY_ORDER = ('Action & Sports', 'Nature & Landscape', 'Urban & City', 'Animals',
                                 'People & Daily Life', 'Weather & Natural Phenomena',
                                 'Vehicles & Transport', 'Art & Performance')

# 安全优先顺序：纯自然/静物在前，人物动作与生活场景在后——prompt_limit 默认取首条，
# 默认配置因此落在零风险样本上；要覆盖度就往下取。
_SAFE_VIDEO_CATEGORY_ORDER = ('Nature & Landscape', 'Weather & Natural Phenomena', 'Animals', 'Vehicles & Transport', 'Urban & City', 'Art & Performance', 'People & Daily Life', 'Action & Sports')

_VIDEO_DATASET_ORDERS = {'msr_vtt': _DEFAULT_VIDEO_CATEGORY_ORDER,
                        'msr_vtt_safe': _SAFE_VIDEO_CATEGORY_ORDER}


def _load_msr_vtt_prompts(limit: int) -> List[str]:
    """历史基线集合（80 条，类别顺序与文本自 2026-10 起冻结）。"""
    return _builtin_video_prompts(_DEFAULT_VIDEO_CATEGORY_ORDER)[:limit]


def _load_video_safe_prompts(limit: int) -> List[str]:
    """安全版集合（80 条，安全场景前置、风险条目等价降级）。"""
    return _builtin_video_prompts(_SAFE_VIDEO_CATEGORY_ORDER, degrade=True)[:limit]
