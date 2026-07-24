"""Audio prompt loader — builtin ASR/TTS datasets."""

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

# Builtin TTS test prompts
TTS_BUILTIN = [
    '你好，欢迎使用语音合成系统。',
    '今天天气真好，适合出去散步。',
    '人工智能正在改变世界。',
    '请确认您的身份信息。',
    'Hello world, this is a test of text to speech.',
    'The quick brown fox jumps over the lazy dog.',
    '一二三四五六七八九十。',
    '请问最近的医院怎么走？',
    '这个产品的质量非常好，我强烈推荐。',
    'Welcome to the speech synthesis evaluation system.',
]


def load_asr_samples(
    dataset: str = 'builtin',
    limit: int = 10,
    custom_path: Optional[str] = None,
) -> List[dict]:
    """Load ASR test samples. Returns list of {text, lang} dicts.

    For ASR, the audio file is uploaded by the user, not provided by the dataset.
    The dataset only provides reference texts for WER/CER comparison.
    """
    if dataset == 'custom' and custom_path:
        return _load_custom_asr(custom_path, limit)

    samples = []
    for ref_text, _, lang in ASR_BUILTIN[:limit]:
        samples.append({'text': ref_text, 'lang': lang})
    return samples


def load_tts_prompts(
    dataset: str = 'builtin',
    limit: int = 10,
    custom_path: Optional[str] = None,
) -> List[str]:
    """Load TTS text prompts. Returns list of strings."""
    if dataset == 'custom' and custom_path:
        return _load_custom_lines(custom_path, limit)
    return TTS_BUILTIN[:limit]


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
