"""Offline request checkpoint regressions; no provider or download access."""
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from evalscope.backend.rag_eval.models.encoder import APIEncoder
from evalscope.backend.rag_eval.models.reranker import APIReranker


@pytest.fixture
def encoder_factory(monkeypatch, tmp_path):
    def make(namespace='identity', enabled=True, **kwargs):
        client = Mock()
        client.embed_documents = Mock(side_effect=lambda texts, **kw: [[float(len(t)), 0.25] for t in texts])
        module = types.ModuleType('langchain_openai.embeddings')
        module.OpenAIEmbeddings = Mock(return_value=client)
        monkeypatch.setitem(sys.modules, 'langchain_openai.embeddings', module)
        model = APIEncoder('embedding-model', 'https://example.invalid/v1', api_key='secret-never-persist',
                           batch_size=2, **kwargs)
        model._supported_encode_params = {'option'}
        if enabled:
            model.request_cache_dir = str(tmp_path)
            model.request_cache_namespace = namespace
        return model, client
    return make


def test_embedding_partial_interrupt_new_instance_resume(encoder_factory, tmp_path):
    model, client = encoder_factory()
    client.embed_documents.side_effect = [[[1., 2.], [3., 4.]], KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        model.encode(['a', 'b', 'c'])
    assert list(tmp_path.rglob('*.npz'))
    resumed, client = encoder_factory()
    actual = resumed.encode(['a', 'b', 'c'])
    assert client.embed_documents.call_count == 1
    assert client.embed_documents.call_args.args[0] == ['c']
    assert torch.equal(actual, torch.tensor([[1., 2.], [3., 4.], [1., .25]]))


@pytest.mark.parametrize('change', ['input', 'namespace', 'dimensions', 'option', 'prompt'])
def test_embedding_request_identity(encoder_factory, change):
    first, _ = encoder_factory()
    first.encode(['a'], option=1)
    warm, warm_client = encoder_factory()
    warm.encode(['a'], option=1)
    assert warm_client.embed_documents.call_count == 0
    second, client = encoder_factory(namespace='other' if change == 'namespace' else 'identity',
                                     dimensions=3 if change == 'dimensions' else None,
                                     prompt='prefix:' if change == 'prompt' else None)
    second.encode(['b' if change == 'input' else 'a'], option=2 if change == 'option' else 1,
                  prompt_type='query' if change == 'prompt' else None)
    assert client.embed_documents.call_count == 1


def test_embedding_corrupt_cache_and_invalid_response(encoder_factory, tmp_path):
    first, _ = encoder_factory()
    first.encode(['a'])
    entries = list(tmp_path.rglob('*.npz'))
    assert len(entries) == 1
    entries[0].write_bytes(b'corrupt')
    second, client = encoder_factory()
    second.encode(['a'])
    assert client.embed_documents.call_count == 1
    third, client = encoder_factory()
    client.embed_documents.return_value = []
    client.embed_documents.side_effect = None
    with pytest.raises(ValueError, match='count mismatch'):
        third.encode(['bad'])
    assert len(list(tmp_path.rglob('*.npz'))) == 1
    assert all(b'secret-never-persist' not in p.read_bytes() for p in tmp_path.rglob('*') if p.is_file())


def test_embedding_without_cache_keeps_calling(encoder_factory, tmp_path):
    model, client = encoder_factory(enabled=False)
    model.encode(['a'])
    model.encode(['a'])
    assert client.embed_documents.call_count == 2
    assert not list(tmp_path.iterdir())


@pytest.fixture
def reranker_factory(tmp_path):
    def make(namespace='identity', enabled=True, **kwargs):
        model = APIReranker('reranker-model', 'https://example.invalid/v1',
                            api_key='secret-never-persist', batch_size=2, **kwargs)
        model.session.post = Mock(side_effect=lambda *a, **kw: response([
            {'index': i, 'relevance_score': len(doc) / 10.}
            for i, doc in enumerate(kw['json'].get('documents', kw['json'].get('input', {}).get('documents', [])))
        ]))
        if enabled:
            model.request_cache_dir = str(tmp_path)
            model.request_cache_namespace = namespace
        return model
    return make


def response(results):
    result = Mock(status_code=200)
    result.json.return_value = {'results': results}
    return result


def test_reranker_partial_interrupt_new_instance_resume(reranker_factory, tmp_path):
    model = reranker_factory()
    model.session.post.side_effect = [response([{'index': 1, 'score': .8}, {'index': 0, 'score': .2}]),
                                      KeyboardInterrupt()]
    pairs = [('q', 'a'), ('q', 'bb'), ('q2', 'ccc')]
    with pytest.raises(KeyboardInterrupt):
        model.predict(pairs)
    assert len(list(tmp_path.rglob('*.npz'))) == 1
    resumed = reranker_factory()
    assert torch.allclose(resumed.predict(pairs), torch.tensor([.2, .8, .3]))
    assert resumed.session.post.call_count == 1
    assert resumed.session.post.call_args.kwargs['json']['query'] == 'q2'


@pytest.mark.parametrize('change', ['input', 'query', 'namespace', 'prompt', 'model', 'endpoint'])
def test_reranker_identity(reranker_factory, change):
    first = reranker_factory()
    first.predict([('q', 'doc')])
    warm = reranker_factory()
    warm.predict([('q', 'doc')])
    assert warm.session.post.call_count == 0
    second = reranker_factory(namespace='other' if change == 'namespace' else 'identity',
                              prompt='prefix' if change == 'prompt' else None)
    if change == 'model':
        second.model_name = 'other-model'
    if change == 'endpoint':
        second.rerank_url = 'https://other.invalid/rerank'
    second.predict([('other' if change == 'query' else 'q', 'other' if change == 'input' else 'doc')])
    assert second.session.post.call_count == 1


def test_reranker_corrupt_failure_and_no_credentials(reranker_factory, tmp_path):
    model = reranker_factory()
    model.predict([('q', 'a')])
    entries = list(tmp_path.rglob('*.npz'))
    assert len(entries) == 1
    entries[0].write_bytes(b'corrupt')
    resumed = reranker_factory()
    resumed.predict([('q', 'a')])
    assert resumed.session.post.call_count == 1
    resumed.session.post.side_effect = None
    resumed.session.post.return_value = Mock(status_code=200, json=Mock(return_value={}))
    with pytest.raises(ValueError, match='Invalid rerank response'):
        resumed.predict([('bad', 'input')])
    assert len(list(tmp_path.rglob('*.npz'))) == 1
    assert all(b'secret-never-persist' not in p.read_bytes() for p in tmp_path.rglob('*') if p.is_file())


def test_reranker_no_cache_and_truncation_semantics(reranker_factory, tmp_path):
    plain = reranker_factory(enabled=False)
    plain.predict([('q', 'doc')])
    plain.predict([('q', 'doc')])
    assert plain.session.post.call_count == 2
    model = reranker_factory(max_input_chars=2)
    expected = model.predict([('query', 'document')])
    resumed = reranker_factory(max_input_chars=2)
    assert torch.equal(resumed.predict([('query', 'document')]), expected)
    assert resumed.session.post.call_count == 0
    assert resumed.input_audit['evaluated_pairs'] == 1
    assert resumed.input_audit['truncated_pairs'] == 1


@pytest.mark.parametrize('corruption', ['object', 'wrong_shape', 'wrong_count'])
def test_embedding_structurally_corrupt_archive_is_miss(encoder_factory, tmp_path, corruption):
    model, _ = encoder_factory()
    model.encode(['a'])
    path, = tmp_path.rglob('*.npz')
    values = {'object': np.array([['unsafe']], dtype=object),
              'wrong_shape': np.array([1.]), 'wrong_count': np.ones((2, 2))}[corruption]
    np.savez(path, values=values)
    resumed, client = encoder_factory()
    assert torch.equal(resumed.encode(['a']), torch.tensor([[1., .25]]))
    assert client.embed_documents.call_count == 1


def test_atomic_replace_failure_leaves_no_partial_checkpoint(encoder_factory, tmp_path, monkeypatch):
    from evalscope.backend.rag_eval.models import request_cache
    model, _ = encoder_factory()
    with monkeypatch.context() as patcher:
        patcher.setattr(request_cache.os, 'replace', Mock(side_effect=OSError('disk failure')))
        assert torch.equal(model.encode(['a']), torch.tensor([[1., .25]]))
    assert not [p for p in tmp_path.rglob('*') if p.is_file()]
    resumed, client = encoder_factory()
    resumed.encode(['a'])
    assert client.embed_documents.call_count == 1


def test_reranker_http_failure_is_retried_on_resume(reranker_factory, tmp_path):
    import requests
    model = reranker_factory()
    failed = Mock(status_code=400, headers={})
    failed.json.return_value = {}
    failed.raise_for_status.side_effect = requests.HTTPError('failure')
    model.session.post.side_effect = None
    model.session.post.return_value = failed
    with pytest.raises(requests.HTTPError):
        model.predict([('q', 'doc')])
    assert not list(tmp_path.rglob('*.npz'))
    resumed = reranker_factory()
    resumed.predict([('q', 'doc')])
    assert resumed.session.post.call_count == 1


@pytest.mark.parametrize('results', [
    [], [{'index': 0, 'score': .1}],
    [{'index': 0, 'score': .1}, {'index': 0, 'score': .2}],
    [{'index': -1, 'score': .1}, {'index': 1, 'score': .2}],
    [{'index': 2, 'score': .1}, {'index': 1, 'score': .2}],
    [{'index': 0}, {'index': 1, 'score': .2}],
    *[[{'index': 0, 'score': value}, {'index': 1, 'score': .2}]
      for value in [float('nan'), float('inf'), -float('inf'), None, 'bad', True]],
    [{'index': .5, 'score': .1}, {'index': 1, 'score': .2}],
    [{'index': True, 'score': .1}, {'index': 0, 'score': .2}],
    ['bad', {'index': 1, 'score': .2}],
], ids=['empty', 'missing', 'duplicate', 'negative', 'out_of_range', 'no_score',
        'nan', 'inf', 'negative_inf', 'null', 'text', 'bool_score', 'fraction_index', 'bool_index', 'not_dict'])
def test_reranker_incomplete_response_never_checkpointed(reranker_factory, tmp_path, results):
    model = reranker_factory()
    model.session.post.side_effect = None
    model.session.post.return_value = response(results)
    with pytest.raises(ValueError, match='Invalid rerank response'):
        model.predict([('q', 'a'), ('q', 'bb')])
    assert not list(tmp_path.rglob('*.npz'))
    resumed = reranker_factory()
    assert torch.allclose(resumed.predict([('q', 'a'), ('q', 'bb')]), torch.tensor([.1, .2]))
    assert resumed.session.post.call_count == 1


def test_reranker_positional_and_uncached_compatibility(reranker_factory):
    model = reranker_factory()
    model.session.post.side_effect = None
    model.session.post.return_value = response([{'score': .2}, {'relevance_score': .4}])
    assert torch.equal(model.predict([('q', 'a'), ('q', 'b')]), torch.tensor([.2, .4]))
    plain = reranker_factory(enabled=False)
    plain.session.post.side_effect = None
    plain.session.post.return_value = response([])
    assert torch.equal(plain.predict([('q', 'a')]), torch.tensor([0.]))


@pytest.mark.parametrize('kind', ['embedding', 'rerank'])
def test_nonfinite_cache_is_a_miss(kind, encoder_factory, reranker_factory, tmp_path):
    if kind == 'embedding':
        original, _ = encoder_factory()
        original.encode(['sensitive-input'])
        damaged = np.array([[np.nan, .25]])
    else:
        original = reranker_factory()
        original.predict([('private-query', 'sensitive-input')])
        damaged = np.array([np.inf])
    path, = tmp_path.rglob('*.npz')
    np.savez(path, values=damaged)
    if kind == 'embedding':
        resumed, client = encoder_factory()
        assert torch.equal(resumed.encode(['sensitive-input']), torch.tensor([[15., .25]]))
        assert client.embed_documents.call_count == 1
    else:
        resumed = reranker_factory()
        assert torch.allclose(resumed.predict([('private-query', 'sensitive-input')]), torch.tensor([1.5]))
        assert resumed.session.post.call_count == 1


@pytest.mark.parametrize('values', [
    [[], []], [[float('nan')], [1.]], [[float('inf')], [1.]],
    [[1.], [2., 3.]], [1., 2.], [['text'], ['text']], [[True], [False]],
])
def test_invalid_embedding_vectors_never_checkpointed(encoder_factory, tmp_path, values):
    model, client = encoder_factory()
    client.embed_documents.side_effect = None
    client.embed_documents.return_value = values
    with pytest.raises(ValueError, match='Invalid embedding response'):
        model.encode(['a', 'b'])
    assert not list(tmp_path.rglob('*.npz'))


def test_embedding_dimension_change_does_not_checkpoint_bad_batch(encoder_factory, tmp_path):
    model, client = encoder_factory()
    client.embed_documents.side_effect = [[[1., 2.], [3., 4.]], [[5.]]]
    with pytest.raises(ValueError, match='Invalid embedding response'):
        model.encode(['a', 'b', 'c'])
    assert len(list(tmp_path.rglob('*.npz'))) == 1


@pytest.mark.parametrize('operation', ['mkdir', 'temporary', 'save', 'fsync', 'replace'])
def test_cache_write_failure_warns_without_sensitive_data(encoder_factory, tmp_path, monkeypatch, caplog, operation):
    import logging
    from pathlib import Path
    from evalscope.backend.rag_eval.models import request_cache

    secret = 'private-query sensitive-input secret-never-persist'
    target, attribute = {
        'mkdir': (Path, 'mkdir'),
        'temporary': (request_cache.tempfile, 'NamedTemporaryFile'),
        'save': (request_cache.np, 'savez_compressed'),
        'fsync': (request_cache.os, 'fsync'),
        'replace': (request_cache.os, 'replace'),
    }[operation]
    model, _ = encoder_factory()
    with caplog.at_level(logging.WARNING), monkeypatch.context() as patcher:
        patcher.setattr(target, attribute, Mock(side_effect=OSError(secret)))
        assert torch.equal(model.encode(['sensitive-input']), torch.tensor([[15., .25]]))
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert warnings, 'Cache write failure must be observable'
    assert any('cache' in record.getMessage().lower() for record in warnings)
    assert all(word not in caplog.text for word in secret.split())
    assert not [p for p in tmp_path.rglob('*') if p.is_file()]


def test_cache_cleanup_failure_keeps_provider_result(encoder_factory, monkeypatch, tmp_path, caplog):
    import logging
    from evalscope.backend.rag_eval.models import request_cache

    model, _ = encoder_factory()
    with caplog.at_level(logging.WARNING), monkeypatch.context() as patcher:
        patcher.setattr(request_cache.os, 'fsync', Mock(side_effect=OSError('private-query secret-never-persist')))
        patcher.setattr(request_cache.os, 'unlink', Mock(side_effect=OSError('sensitive-input secret-never-persist')))
        assert torch.equal(model.encode(['sensitive-input']), torch.tensor([[15., .25]]))
    assert 'cache' in caplog.text.lower()
    assert all(word not in caplog.text for word in ['private-query', 'sensitive-input', 'secret-never-persist'])
    assert not list(tmp_path.rglob('*.npz'))


def test_empty_dimension_embedding_cache_is_a_miss(encoder_factory, tmp_path):
    model, _ = encoder_factory()
    model.encode(['a'])
    path, = tmp_path.rglob('*.npz')
    np.savez(path, values=np.empty((1, 0)))
    resumed, client = encoder_factory()
    assert torch.equal(resumed.encode(['a']), torch.tensor([[1., .25]]))
    assert client.embed_documents.call_count == 1
