import pytest

from evalscope.backend.rag_eval.models.encoder import (
    DEFAULT_API_BATCH_SIZE,
    MAX_API_BATCH_SIZE,
    _effective_api_batch_size,
    _validate_embedding_response_count,
)


def test_api_batch_size_policy_uses_project_default_and_cap():
    assert DEFAULT_API_BATCH_SIZE == 20
    assert MAX_API_BATCH_SIZE == 20
    assert _effective_api_batch_size(None) == 20
    assert _effective_api_batch_size(1) == 1
    assert _effective_api_batch_size(20) == 20
    assert _effective_api_batch_size(30) == 20


def test_embedding_response_count_must_match_request_count():
    _validate_embedding_response_count(['a', 'b'], [[1], [2]])
    with pytest.raises(ValueError, match='Embedding response count mismatch'):
        _validate_embedding_response_count(['a', 'b'], [[1]])
