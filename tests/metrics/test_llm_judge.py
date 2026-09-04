from unittest import mock

import pytest

from evalscope.metrics.llm_judge import LLMJudge, LLMJudgeError, LLMJudgeParseError


def _make_judge(model):
    """Build an LLMJudge without network/model init, injecting ``model``."""
    with mock.patch.object(LLMJudge, '_init_server_adapter', return_value=None):
        judge = LLMJudge(
            model_id='fake-judge',
            api_url='http://fake.invalid/v1',
            api_key='test-key',
            max_retries=2,
            retry_backoff=0.0,
        )
    judge.model = model
    return judge


def test_judge_transport_failure_retries_then_raises():
    """A failing judge backend must raise LLMJudgeError after bounded retries,
    not return an '[ERROR]' string that downstream silently scores as 0."""
    model = mock.Mock()
    model.generate.side_effect = RuntimeError('judge API down')

    judge = _make_judge(model)

    with pytest.raises(LLMJudgeError):
        judge.judge(prompt='test')

    assert model.generate.call_count == 3  # max_retries(2) + 1


def test_judge_empty_completion_raises():
    """An empty judge completion is a judge failure, not a zero score."""
    model = mock.Mock()
    model.generate.return_value = mock.Mock(completion=None)

    judge = _make_judge(model)

    with pytest.raises(LLMJudgeError):
        judge.judge(prompt='test')


def test_get_score_none_raises_parse_error():
    """A None judge response must raise LLMJudgeParseError, not silently return 0.0."""
    judge = _make_judge(mock.Mock())

    with pytest.raises(LLMJudgeParseError):
        judge.get_score(None)  # type: ignore[arg-type]
