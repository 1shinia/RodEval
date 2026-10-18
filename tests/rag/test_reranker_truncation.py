"""Explicit character truncation must remain opt-in and auditable."""
import json
import pytest
import requests
from evalscope.backend.rag_eval.models.reranker import APIReranker


@pytest.mark.parametrize('limit', [None, 4])
def test_policy_payload_and_audit(monkeypatch, tmp_path, limit):
    model = APIReranker('m', 'https://example.invalid/v1', max_input_chars=limit)
    model.progress_path = str(tmp_path / 'progress.json')
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs['json'])
        r = requests.Response()
        r.status_code = 200
        r._content = b'{"results":[{"index":0,"relevance_score":0.5}]}'
        return r
    monkeypatch.setattr(model.session, 'post', post)
    model.predict([('query-long', 'document-long')])
    assert sent[0]['query'] == ('quer' if limit else 'query-long')
    assert sent[0]['documents'] == (['docu'] if limit else ['document-long'])
    if limit:
        audit = json.loads((tmp_path / 'reranker_input_policy.json').read_text())
        assert audit['max_input_chars'] == 4
        assert audit['truncated_pairs'] == 1
        assert audit['evaluated_pairs'] == 1
        assert audit['unit'] == 'unicode_characters'
    else:
        assert not (tmp_path / 'reranker_input_policy.json').exists()


@pytest.mark.parametrize('limit', [0, -1, True, 1.5, '4'])
def test_invalid_policy_rejected(limit):
    with pytest.raises(ValueError, match='max_input_chars'):
        APIReranker('m', 'https://example.invalid/v1', max_input_chars=limit)
