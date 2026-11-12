"""Audio prompt loader — builtin ASR/TTS datasets."""

import json
import random
from pathlib import Path
from typing import List, Optional

# Builtin ASR test samples: (audio_filename_hint, reference_text, language)
ASR_BUILTIN = [
    ('今天天气真好，适合出去散步。', '今天天气真好，适合出去散步。', 'zh'),
    ('你好，请问最近的医院怎么走？', '你好，请问最近的医院怎么走？', 'zh'),
    ('欢迎使用语音识别评估系统。', '欢迎使用语音识别评估系统。', 'zh'),
    ('这个模型的效果还不错，准确率很高。', '这个模型的效果还不错，准确率很高。', 'zh'),
    ('请说一段话以便测试系统的性能。', '请说一段话以便测试系统的性能。', 'zh'),
    ('hello world this is a test', 'hello world this is a test', 'en'),
    ('the quick brown fox jumps over the lazy dog', 'the quick brown fox jumps over the lazy dog', 'en'),
    ('speech recognition is an important technology', 'speech recognition is an important technology', 'en'),
    ('人工智能正在改变我们的生活', '人工智能正在改变我们的生活', 'zh'),
    ('深度学习是机器学习的一个重要分支', '深度学习是机器学习的一个重要分支', 'zh'),
]

# One source of truth for the indexed, categorized TTS test texts.
TTS_DATASET = json.loads(Path(__file__).with_name('tts_basic_100_v1.json').read_text(encoding='utf-8'))
TTS_BUILTIN = [item['text'] for item in TTS_DATASET['items']]


def load_asr_samples(
    dataset: str = 'builtin',
    limit: int = 10,
    custom_path: Optional[str] = None,
) -> List[dict]:
    """Load ASR test samples. Returns list of {text, lang} dicts.

    For ASR, the audio file is uploaded by the user, not provided by the dataset.
    The dataset only provides reference texts for WER/CER comparison.
    """
    if dataset == 'custom':
        if not custom_path:
            raise ValueError('custom ASR dataset path is required')
        return _load_custom_asr(custom_path, limit)

    samples = []
    picked = random.sample(ASR_BUILTIN, min(limit, len(ASR_BUILTIN)))
    for ref_text, _, lang in picked:
        samples.append({'text': ref_text, 'lang': lang})
    return samples


def load_tts_prompts(
    dataset: str = 'builtin',
    limit: int = 10,
    custom_path: Optional[str] = None,
    seed: int = 42,
    selected_index: Optional[int] = None,
    category: Optional[str] = None,
) -> List[str]:
    """Load TTS text prompts. Returns list of strings."""
    if dataset == 'custom':
        if not custom_path:
            raise ValueError('custom TTS dataset path is required')
        return _load_custom_lines(custom_path, limit)
    items = TTS_DATASET['items']
    if category:
        items = [item for item in items if item['category'] == category]
        if not items:
            raise ValueError('Unknown TTS prompt category')
    if selected_index is not None:
        if not 0 <= selected_index < len(TTS_BUILTIN):
            raise ValueError('TTS prompt index is out of range')
        selected = TTS_DATASET['items'][selected_index]
        if category and selected['category'] != category:
            raise ValueError('Selected TTS prompt is outside the category')
        return [selected['text']]
    return random.Random(seed).sample([item['text'] for item in items], min(limit, len(items)))


def _load_custom_asr(path: str, limit: int) -> List[dict]:
    """Load ASR reference texts from file. Format: text|lang per line."""
    lines = _read_lines(path, limit)
    samples = []
    for line in lines:
        if '|' in line:
            text, lang = line.rsplit('|', 1)
            samples.append({'text': text.strip(), 'lang': lang.strip()})
        else:
            samples.append({'text': line, 'lang': 'zh'})
    return samples


def _load_custom_lines(path: str, limit: int) -> List[str]:
    """Load simple text lines from file (for TTS)."""
    return _read_lines(path, limit)


def _read_lines(path: str, limit: int) -> List[str]:
    """Read lines from a text file, skipping comments and blanks."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}')

    with open(p, 'r') as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]
    return lines[:limit]
