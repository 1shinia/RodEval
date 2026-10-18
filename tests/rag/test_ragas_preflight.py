"""RAGAS evaluation-only preflight and model configuration regressions."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evalscope.backend.rag_eval.ragas import task_template as module
from evalscope.backend.rag_eval.ragas.arguments import RAGASEvalConfig, RAGASLLMConfig


def test_llm_generation_options_are_forwarded():
    config = RAGASLLMConfig(model_name='offline', api_key='offline-placeholder',
                            api_base='http://127.0.0.1:1/v1', temperature=0.37, max_tokens=123)
    with patch('ragas.llms.llm_factory', return_value=object()) as factory:
        module._build_llm(config)
    assert factory.call_args.kwargs['temperature'] == 0.37
    assert factory.call_args.kwargs['max_tokens'] == 123


@pytest.mark.parametrize('changes', [
    {'metrics': []}, {'metrics': ['context_relevancy']}, {'metrics': ['pathlib.Path']},
    {'critic_llm': {'model_name': 'offline', 'provider': 'unsupported'}},
    {'critic_llm': {'model_name': '   '}},
    {'embeddings': {'model_name_or_path': 'offline', 'provider': 'unsupported'}},
    {'embeddings': {'model_name_or_path': '   ', 'provider': 'openai'}},
    {'metrics': ['answer_relevancy']}, {'batch_size': 0}, {'batch_size': -1},
])
def test_invalid_evaluation_config_is_rejected(changes):
    config = {'testset_file': 'input.json', 'critic_llm': {'model_name': 'offline'},
              'metrics': ['faithfulness']}
    config.update(changes)
    with pytest.raises(ValueError):
        RAGASEvalConfig(**config)


def test_legacy_metric_names_and_language_remain_compatible():
    args = RAGASEvalConfig(testset_file='input.json', critic_llm={'model_name': 'offline'},
                           metrics=['Faithfulness', 'faithfulness'], language='chinese')
    assert args.metrics == ['faithfulness']
    assert args.language == 'chinese'


def test_invalid_embedding_is_rejected_before_llm_initialization(tmp_path, monkeypatch):
    path = tmp_path / 'data.json'
    path.write_text(json.dumps([{'user_input': 'q', 'response': 'a'}]))
    args = SimpleNamespace(testset_file=str(path), metrics=['answer_relevancy'],
                           critic_llm=None, embeddings=SimpleNamespace(
                               provider='unsupported', model_name_or_path='offline'))
    calls = []
    monkeypatch.setattr(module, '_build_llm', lambda _: calls.append('llm'))
    with pytest.raises(ValueError, match='provider'):
        asyncio.run(module._run_ragas_eval(args))
    assert calls == []
