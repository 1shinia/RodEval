"""Offline RAGAS scoring regressions; no model requests."""
import pytest
from evalscope.backend.rag_eval.ragas.task_template import import_metric


def test_metric_instances_are_task_local():
    assert import_metric('faithfulness') is not import_metric('faithfulness')


@pytest.mark.parametrize('name', ['context_relevancy', 'pathlib.Path', 'os.system'])
def test_only_supported_metrics(name):
    with pytest.raises(ValueError, match='Unsupported'):
        import_metric(name)


def test_legacy_metric_class_name():
    assert import_metric('AnswerRelevancy').name == 'answer_relevancy'


def test_remote_embedding_matches_selected_metric(monkeypatch):
    from types import SimpleNamespace
    from evalscope.backend.rag_eval.ragas.arguments import RAGASEmbeddingConfig
    from evalscope.backend.rag_eval.ragas.task_template import _build_embeddings
    from openai.resources.embeddings import Embeddings
    def create(self, **kwargs):
        return {'data': [{'embedding': [1.0, 0.0], 'index': i}
                         for i, _ in enumerate(kwargs['input'])]}
    monkeypatch.setattr(Embeddings, 'create', create)
    model = _build_embeddings(RAGASEmbeddingConfig(
        model_name_or_path='offline', provider='openai', api_key='offline-placeholder',
        api_base='http://127.0.0.1:1/v1'))
    metric = import_metric('answer_relevancy')
    metric.embeddings = model
    assert metric.calculate_similarity('question', ['question'])[0] == pytest.approx(1.0)


@pytest.mark.parametrize('suffix', ['.json', '.jsonl'])
def test_dataset_aliases_and_required_fields(tmp_path, suffix):
    import json
    from evalscope.backend.rag_eval.ragas.task_template import _load_dataset
    row = {'question': 'q', 'answer': 'a', 'contexts': 'c'}
    path = tmp_path / ('input' + suffix)
    path.write_text(json.dumps([row] if suffix == '.json' else row))
    data = _load_dataset(str(path), metrics=[import_metric('faithfulness')])
    assert data.to_list()[0]['retrieved_contexts'] == ['c']
    path.write_text(json.dumps([{'question': 'q'}] if suffix == '.json' else {'question': 'q'}))
    with pytest.raises(ValueError, match='row 1.*response'):
        _load_dataset(str(path), metrics=[import_metric('faithfulness')])


@pytest.mark.parametrize('content', ['[]', 'null', '[42]'])
def test_invalid_dataset(tmp_path, content):
    from evalscope.backend.rag_eval.ragas.task_template import _load_dataset
    path = tmp_path / 'input.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        _load_dataset(str(path))


def test_preflight_and_embedding_dependency(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from evalscope.backend.rag_eval.ragas import task_template as module
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'user_input': 'q', 'response': 'a', 'retrieved_contexts': ['c']}]))
    calls = []
    monkeypatch.setattr(module, '_build_llm', lambda _: calls.append('llm') or object())
    monkeypatch.setattr(module, '_build_embeddings', lambda _: calls.append('embeddings') or object())
    async def evaluate(**kwargs):
        assert kwargs['embeddings'] is None
        return 'ok'
    monkeypatch.setattr('ragas.aevaluate', evaluate)
    args = SimpleNamespace(testset_file=str(path), metrics=['faithfulness'], critic_llm=None,
                           embeddings=None, raise_exceptions=False, batch_size=None)
    assert asyncio.run(module._run_ragas_eval(args)) == 'ok'
    assert calls == ['llm']
    calls.clear()
    args.metrics = []
    with pytest.raises(ValueError, match='metric'):
        asyncio.run(module._run_ragas_eval(args))
    assert calls == []
    args.metrics = ['faithfulness']
    path.write_text('[]')
    with pytest.raises(ValueError):
        asyncio.run(module._run_ragas_eval(args))
    assert calls == []


@pytest.mark.parametrize('scores,expected', [([float('nan'), float('nan')], None), ([0.8, float('nan')], 0.8), ([0.8, 0.6], 0.7)])
def test_score_validity_and_counts(tmp_path, monkeypatch, caplog, scores, expected):
    import pandas as pd
    from types import SimpleNamespace
    from evalscope.backend.rag_eval.ragas import task_template as module
    async def run(_):
        return SimpleNamespace(to_pandas=lambda: pd.DataFrame({'faithfulness': scores}))
    monkeypatch.setattr(module, '_run_ragas_eval', run)
    args = SimpleNamespace(testset_file=str(tmp_path / 'input.json'), metrics=['faithfulness'])
    if expected is None:
        with pytest.raises(ValueError, match='No valid'):
            module.rag_eval(args)
    else:
        with caplog.at_level('INFO'):
            result = module.rag_eval(args)
        assert result['faithfulness'] == pytest.approx(expected)
        assert 'valid=' in caplog.text and 'failed=' in caplog.text


def test_embedding_is_optional_in_faithfulness_config():
    from evalscope.backend.rag_eval.ragas.arguments import RAGASEvalConfig
    config = RAGASEvalConfig(testset_file='input.json', critic_llm={'model_name': 'offline'}, metrics=['faithfulness'])
    assert config.embeddings is None


@pytest.mark.parametrize('contexts', [None, [], [''], ['   ']])
def test_blank_context_rejected(tmp_path, contexts):
    import json
    from evalscope.backend.rag_eval.ragas.task_template import _load_dataset
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([{'user_input': 'q', 'response': 'a', 'retrieved_contexts': contexts}]))
    with pytest.raises(ValueError, match='row 1.*retrieved_contexts'):
        _load_dataset(str(path), metrics=[import_metric('faithfulness')])
