"""WER / CER computation using jiwer."""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

# Chinese + English punctuation to strip before WER/CER
_PUNCT_RE = re.compile(r'[，。！？、；：""''（）《》【】\s,.!?;:\"\'()\[\]{}]')

# Detect CJK (Chinese/Japanese/Korean) characters
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')


def _wordify(text: str) -> str:
    """Split CJK text into characters for word-level metrics.

    jiwer's wer() expects space-separated words. Chinese has no spaces,
    so we insert spaces between CJK characters to treat each character
    as a "word" for WER computation.
    """
    if _CJK_RE.search(text):
        result = []
        for ch in text:
            if _CJK_RE.match(ch):
                result.append(f' {ch} ')
            else:
                result.append(ch)
        text = re.sub(r'\s+', ' ', ''.join(result)).strip()
    return text


def _normalize(text: str) -> str:
    """Strip punctuation for fair WER/CER comparison."""
    return _PUNCT_RE.sub('', text)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate.

    Returns a value >= 0. 0 means perfect match.
    For Chinese text, characters are treated as words.
    """
    try:
        from jiwer import wer
    except ImportError:
        logger.error('jiwer not installed. Install with: pip install jiwer')
        return -1.0

    ref = _wordify(_normalize(reference))
    hyp = _wordify(_normalize(hypothesis))

    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0

    try:
        return float(wer(ref, hyp))
    except Exception as e:
        logger.warning(f'WER computation failed: {e}')
        return -1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (better for Chinese).

    Returns a value >= 0. 0 means perfect match.
    """
    try:
        from jiwer import cer
    except ImportError:
        logger.error('jiwer not installed. Install with: pip install jiwer')
        return -1.0

    ref = _normalize(reference)
    hyp = _normalize(hypothesis)

    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0

    try:
        return float(cer(ref, hyp))
    except Exception as e:
        logger.warning(f'CER computation failed: {e}')
        return -1.0
