"""WER / CER computation using jiwer."""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

# Chinese + English punctuation to strip before WER/CER
_PUNCT_RE = re.compile(r"[，。！？、；：\"'（）《》【】,.!?;:()\[\]{}]")

# Detect CJK (Chinese/Japanese/Korean) characters
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')

# Chinese digit characters mapped to Arabic digits (character-level fallback)
_CN_DIGIT_MAP = {
    '零': '0', '〇': '0',
    '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
    '六': '6', '七': '7', '八': '8', '九': '9',
    '幺': '1',  # phone-number colloquial for 一
}
# Arabic digit → Chinese digit (for digit-by-digit reading normalization)
# Note: map to 零 (not 〇) so that 一零 → 十 replacement works correctly
_ARABIC_TO_CN_DIGIT = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
}
# A run of 2+ consecutive CN digit characters (excluding multiplicative
# characters 十/百/千/万/亿) that cn2an refuses to parse as a formal number.
_CN_DIGIT_RUN_RE = re.compile(r'[零〇一二三四五六七八九幺]{2,}')
# Space between two Chinese digits (for iterative collapse)
_CN_DIGIT_SPACE_RE = re.compile(r'([零一二三四五六七八九])\s+([零一二三四五六七八九])')


def _fullwidth_to_halfwidth(text: str) -> str:
    """Map full-width ASCII letters/digits/space to their half-width forms."""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(' ')
        elif 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return ''.join(out)


def _normalize_chinese_numbers(text: str) -> str:
    """Normalize Chinese/Arabic number expressions for fair comparison.

    Two-phase approach:
      1. cn2an handles formal numeric semantics (一百二十三 → 123,
         百分之五十 → 50%) and protects idioms like 千方百计.
      2. Digit-by-digit normalization unifies "digit sequence reading"
         (一二三四五) and "numeric value reading" (12345 → 一二三四五)
         into the same Chinese digit sequence form. Arabic digits are
         mapped to Chinese digits, spaces between Chinese digits are
         collapsed, and 一零 is mapped to 十.

    The key insight: in speech evaluation, ASR outputs like
    "1 2 3 4 5 6 7 8 9 10" represent digit-by-digit reading, not
    numeric values. Both sides (reference and hypothesis) are
    normalized identically, so semantic equivalence is preserved
    regardless of which written form each side uses.

    If cn2an is unavailable, only the character-level fallback runs.
    """
    try:
        import cn2an
        # cn2an logs UserWarning for malformed numbers; suppress the noise.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            try:
                text = cn2an.transform(text, 'cn2an')
            except Exception:
                # Fall through to the character-level fallback below.
                pass
    except ImportError:
        pass

    # Phase 2: digit-by-digit normalization to Chinese digit sequence
    # Map Arabic digits to Chinese digits (0→零, 1→一, ..., 9→九)
    text = ''.join(_ARABIC_TO_CN_DIGIT.get(c, c) for c in text)
    # Collapse spaces between Chinese digits (一 二 三 → 一二三)
    prev = None
    while prev != text:
        prev = text
        text = _CN_DIGIT_SPACE_RE.sub(r'\1\2', text)
    # Map 一零 to 十 (handles "1 0" → "一零" → "十")
    text = text.replace('一零', '十')

    # Fallback: any remaining runs of Chinese digit characters that
    # cn2an rejected (e.g. 一二三四五六七八九十) are kept as-is in
    # Chinese form, which is the canonical form for comparison.
    return text


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
    """Normalize text for fair WER/CER comparison.

    Steps (order matters):
      1. Map full-width ASCII (０-９Ａ-Ｚａ-ｚ) to half-width, so that
         subsequent number normalization can recognize digits.
      2. Normalize Chinese/Arabic number expressions (一二三四五 → 一二三四五,
         12345 → 一二三四五, 一百二十三 → 一二三, 百分之五十 → 五零%).
      3. Lowercase ASCII letters (Hello → hello).
      4. Replace punctuation with word-separating spaces, collapse whitespace.
    """
    text = _fullwidth_to_halfwidth(text)
    text = _normalize_chinese_numbers(text)
    text = text.lower()
    return re.sub(r'\s+', ' ', _PUNCT_RE.sub(' ', text)).strip()


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

    ref = _normalize(reference).replace(' ', '')
    hyp = _normalize(hypothesis).replace(' ', '')

    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0

    try:
        return float(cer(ref, hyp))
    except Exception as e:
        logger.warning(f'CER computation failed: {e}')
        return -1.0
