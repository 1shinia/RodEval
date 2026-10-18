"""Offline regression tests for safe reranker API diagnostics."""
import json

import pytest
import requests

from evalscope.backend.rag_eval.models.reranker import APIReranker


def response(status, body):
    result = requests.Response()
    result.status_code = status
    result.url = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'
    result._content = body.encode()
    return result


def test_native_error_exposes_reason_without_credentials_or_input(monkeypatch):
    model = APIReranker('qwen3-rerank', 'https://dashscope.aliyuncs.com/api/v1', api_key='private-test-credential')
    reply = response(400, json.dumps({
        'code': 'InvalidParameter',
        'message': 'Input too long: private-query private-document Bearer private-test-credential',
        'request_id': 'request-test-123',
    }))
    monkeypatch.setattr(model.session, 'post', lambda *a, **kw: reply)
    with pytest.raises(requests.HTTPError) as error:
        model.predict([('private-query', 'private-document')])
    text = str(error.value)
    assert 'InvalidParameter' in text
    assert 'Input too long' in text
    assert 'request-test-123' in text
    assert all(secret not in text for secret in ['private-query', 'private-document', 'private-test-credential'])
    assert error.value.response is reply


def test_non_json_error_never_dumps_body(monkeypatch):
    model = APIReranker('model', 'https://example.invalid/v1', api_key='test-secret')
    reply = response(400, 'private proxy HTML containing secrets')
    monkeypatch.setattr(model.session, 'post', lambda *a, **kw: reply)
    with pytest.raises(requests.HTTPError) as error:
        model.predict([('query', 'document')])
    assert '400' in str(error.value)
    assert 'private proxy' not in str(error.value)


def test_success_scores_unchanged(monkeypatch):
    model = APIReranker('qwen3-rerank', 'https://dashscope.aliyuncs.com/api/v1')
    reply = response(200, json.dumps({'output': {'results': [{'index': 1, 'relevance_score': 0.25}, {'index': 0, 'relevance_score': 0.75}]}}))
    monkeypatch.setattr(model.session, 'post', lambda *a, **kw: reply)
    assert model.predict([('q', 'first'), ('q', 'second')]).tolist() == [0.75, 0.25]
