"""Real MTEB scoring/cache with local data and a provider-only replacement."""
import sys
import types
from unittest.mock import Mock

import mteb
import pytest
from datasets import Dataset
from mteb.abstasks.sts import AbsTaskSTS
from mteb.tasks.sts.eng.sts12_sts import STS12STS

from evalscope.backend.rag_eval.models.encoder import APIEncoder
from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig, MTEBModelConfig
from evalscope.backend.rag_eval.mteb.task_template import _configure_resume, _build_evaluate_kwargs


class LocalSTS(AbsTaskSTS):
    metadata = STS12STS.metadata.model_copy(update={
        'name': 'ResumeLocalSTS', 'dataset': {'path': 'local-only', 'revision': 'audit-v1'}})

    def load_data(self, **kwargs):
        self.dataset = {'default': {'test': Dataset.from_dict({
            'sentence1': ['a', 'bb', 'ccc', 'dddd'],
            'sentence2': ['ee', 'fff', 'gggg', 'h'], 'score': [1., 4., 3., 0.]})}}
        self.data_loaded = True


def make_model(monkeypatch, root, interrupt=False):
    calls = []

    def embed(texts, **kwargs):
        calls.append(list(texts))
        if interrupt and len(calls) == 2:
            raise KeyboardInterrupt('local interruption')
        return [[float(len(text)), float(ord(text[0]) % 7 + 1), 1.] for text in texts]

    client = Mock()
    client.embed_documents = embed
    module = types.ModuleType('langchain_openai.embeddings')
    module.OpenAIEmbeddings = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, 'langchain_openai.embeddings', module)
    model = APIEncoder('audit', 'https://example.invalid/v1', api_key='local', batch_size=2)
    config = MTEBModelConfig(model_name='audit', api_base='https://example.invalid/v1')
    args = MTEBEvalConfig(output_folder=str(root), task_names=['ResumeLocalSTS'])
    checkpoint = _configure_resume(model, config, args)
    return model, calls, _build_evaluate_kwargs(args, checkpoint)


def run(model, kwargs):
    return mteb.evaluate(model, [LocalSTS()], show_progress_bar=False, **kwargs)


def test_real_mteb_completed_result_avoids_all_provider_calls(tmp_path, monkeypatch):
    model, calls, kwargs = make_model(monkeypatch, tmp_path)
    first = run(model, kwargs)
    assert calls
    restored, calls, kwargs = make_model(monkeypatch, tmp_path)
    second = run(restored, kwargs)
    assert calls == []
    # MTEB serializes cached numeric scores to six decimal places.
    for actual, expected in zip(second.task_results[0].scores['test'], first.task_results[0].scores['test']):
        for name, value in expected.items():
            if isinstance(value, (int, float)):
                assert actual[name] == pytest.approx(value, abs=1e-6)
            else:
                assert actual[name] == value


def test_real_mteb_interrupted_requests_resume_with_same_scores(tmp_path, monkeypatch):
    model, interrupted, kwargs = make_model(monkeypatch, tmp_path / 'resume', interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        run(model, kwargs)
    restored, resumed_calls, kwargs = make_model(monkeypatch, tmp_path / 'resume')
    resumed = run(restored, kwargs)
    baseline, full_calls, kwargs = make_model(monkeypatch, tmp_path / 'baseline')
    full = run(baseline, kwargs)
    assert interrupted[0] not in resumed_calls
    assert len(resumed_calls) == len(full_calls) - 1
    assert resumed.task_results[0].scores == full.task_results[0].scores
