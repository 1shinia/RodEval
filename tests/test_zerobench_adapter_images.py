"""Regression test: zerobench adapter must not crash on alpha-bearing PIL images.

Bug: record_to_sample() handed PIL Images with alpha (RGBA/LA/P) straight to
img.save(buf, format='JPEG'), raising "OSError: cannot write mode RGBA as JPEG".
The original 1-sample smoke happened to miss the offending image; raising the
limit to 20 blew it up. Fix flattens alpha to RGB before saving as JPEG.
"""
import io
import sys

import pytest
from PIL import Image

sys.path.insert(0, '/data/proj/eval/evalperf')


def _make_record(mode):
    """Minimal zerobench record carrying one PIL image in the given mode."""
    return {
        'question_text': 'How many red squares?',
        'question_images_decoded': [Image.new(mode, (64, 64))],
        'question_id': 'probe-rgba',
        'question_images': ['probe.png'],
        'image_attribution': 'probe',
        'question_answer': '4',
    }


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P', 'RGB', 'L'])
def test_record_to_sample_handles_alpha_images(mode):
    """No mode should raise on the JPEG fallback path (format=None -> JPEG)."""
    from evalscope.api.registry import get_benchmark
    from evalscope.service.blueprints.eval import _build_task_config_openai

    cfg = _build_task_config_openai({
        'model': 'dummy', 'api_url': 'https://example/v1/chat/completions',
        'api_key': 'x', 'eval_type': 'openai_api', 'datasets': ['zerobench'],
        'dataset_args': {'zerobench': {}}, 'eval_batch_size': 1, 'limit': 1,
        'timeout': 300, 'generation_config': {'max_tokens': 8192},
        'work_dir': '/tmp/zb_regression', 'sandbox': {'enabled': False},
    })
    bm = get_benchmark('zerobench')
    bm._task_config = cfg

    sample = bm.record_to_sample(_make_record(mode))
    # A text content + at least one image content must be produced.
    content = sample.input[0].content
    assert len(content) >= 2
    assert any(getattr(c, 'image', None) for c in content)


def test_alpha_image_would_fail_without_fix():
    """Guard the guard: confirm the raw PIL call really does raise for RGBA.

    If PIL ever stops raising here, this test tells us the fix became moot
    rather than silently passing.
    """
    buf = io.BytesIO()
    with pytest.raises(OSError):
        Image.new('RGBA', (8, 8)).save(buf, format='JPEG')