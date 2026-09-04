"""Unit tests for the per-sample progress log id fallback chain
(moved from the fragile metadata('instance_id', metadata('id', '?')) pattern
to instance_id → native Sample.id → subset index)."""

import logging

from evalscope.api.dataset import Sample
from evalscope.evaluator.evaluator import DefaultEvaluator, _WorkItem


class _FakeBenchmark:
    """Non-batch benchmark (predict+review atomic per sample), like GSM8K."""
    use_batch_scoring = False


class _FakeTaskState:
    pass


class _FakeSampleScore:
    pass


def _make_evaluator(monkeypatch):
    ev = DefaultEvaluator.__new__(DefaultEvaluator)
    ev.model_name = 'test-model'  # type: ignore[attr-defined]
    ev.benchmark = _FakeBenchmark()  # type: ignore[attr-defined]
    # Drive the real _process_work_item through the predict path, but stub the
    # heavy compute (inference + review) so we reach the log branch cheaply.
    monkeypatch.setattr(ev, '_predict_sample', lambda sample, d: _FakeTaskState())
    monkeypatch.setattr(ev, '_review_task_state', lambda ts: _FakeSampleScore())
    return ev


def _run_capture(monkeypatch, caplog, item):
    ev = _make_evaluator(monkeypatch)
    with caplog.at_level(logging.INFO, logger='evalperf'):
        ev._process_work_item(item, '/tmp/mpd')  # type: ignore[attr-defined]
    matches = [r.message for r in caplog.records
               if 'Prediction done for sample' in r.message]
    assert matches, f"log line not emitted; records={[r.message for r in caplog.records]}"
    return matches[0]


def test_instance_id_wins_over_native_id(monkeypatch, caplog):
    item = _WorkItem(
        subset='main',
        sample=Sample(input='q', target='a', id=7,
                      metadata={'instance_id': 'swe-bench__django-1234'}),
        sample_idx=0,
    )
    msg = _run_capture(monkeypatch, caplog, item)
    assert 'swe-bench__django-1234' in msg
    assert '#0' not in msg


def test_native_id_used_when_no_instance_id(monkeypatch, caplog):
    item = _WorkItem(
        subset='main',
        sample=Sample(input='q', target='t', id=42, metadata={'other': 'x'}),
        sample_idx=0,
    )
    msg = _run_capture(monkeypatch, caplog, item)
    assert '42' in msg and '#0' not in msg


def test_plain_text_dataset_falls_back_to_index(monkeypatch, caplog):
    item = _WorkItem(
        subset='main',
        sample=Sample(input='q', target='t'),
        sample_idx=3,
    )
    msg = _run_capture(monkeypatch, caplog, item)
    assert '#3' in msg
    assert '?' not in msg


def test_no_fallback_question_mark_ever(monkeypatch, caplog):
    """Regression: the old code emitted '?' for idless samples."""
    item = _WorkItem(
        subset='main',
        sample=Sample(input='q', target='t'),
        sample_idx=0,
    )
    msg = _run_capture(monkeypatch, caplog, item)
    assert '?' not in msg