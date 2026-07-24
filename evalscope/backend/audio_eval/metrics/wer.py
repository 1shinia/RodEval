"""WER / CER computation using jiwer."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate.

    Returns a value >= 0. 0 means perfect match.
    """
    try:
        from jiwer import wer
    except ImportError:
        logger.error('jiwer not installed. Install with: pip install jiwer')
        return -1.0

    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0

    try:
        return float(wer(reference, hypothesis))
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

    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0

    try:
        return float(cer(reference, hypothesis))
    except Exception as e:
        logger.warning(f'CER computation failed: {e}')
        return -1.0
