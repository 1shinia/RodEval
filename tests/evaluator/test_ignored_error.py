from evalscope.api.dataset import Sample
from evalscope.evaluator.evaluator import DefaultEvaluator, _WorkItem, _WorkItemProcessingError


class _FakeBenchmark:
    metric_list = ['acc']


def _make_evaluator():
    ev = DefaultEvaluator.__new__(DefaultEvaluator)
    ev.model_name = 'test-model'  # type: ignore[attr-defined]
    ev.benchmark = _FakeBenchmark()  # type: ignore[attr-defined]
    return ev


def _make_item():
    return _WorkItem(subset='main', sample=Sample(input='q', target='42'))


def test_scoring_failure_produces_empty_score_not_zero():
    """Judge/scorer infrastructure failure must not be recorded as a zero score;
    it becomes an empty metric so aggregators expose a coverage gap."""
    ev = _make_evaluator()
    item = _make_item()
    exc = _WorkItemProcessingError('scoring', RuntimeError('judge down'))

    _, sample_score = ev._build_ignored_error_result(item, exc)  # type: ignore[attr-defined]

    assert sample_score.score.value == {}
    assert sample_score.score.metadata['status'] == 'scoring_error'


def test_inference_failure_produces_zero_score():
    """Model/inference failure counts as zero for configured model-quality metrics."""
    ev = _make_evaluator()
    item = _make_item()
    exc = _WorkItemProcessingError('inference', RuntimeError('model down'))

    _, sample_score = ev._build_ignored_error_result(item, exc)  # type: ignore[attr-defined]

    assert sample_score.score.value == {'acc': 0.0}
    assert sample_score.score.metadata['status'] == 'model_error'


def test_unknown_stage_produces_empty_score():
    """Unclassified failures are treated conservatively as empty scores."""
    ev = _make_evaluator()
    item = _make_item()
    exc = _WorkItemProcessingError('unknown', RuntimeError('boom'))

    _, sample_score = ev._build_ignored_error_result(item, exc)  # type: ignore[attr-defined]

    assert sample_score.score.value == {}
    assert sample_score.score.metadata['status'] == 'processing_error'
