"""Offline regressions for managed result publication and stage-one predictions."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalscope.backend.rag_eval.mteb import task_template as runner
from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig, MTEBModelConfig


def result(checkpoint, name):
    path = Path(checkpoint) / 'results' / name / 'master' / 'task.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"task_name":"task","scores":{}}')
    return path


def test_publication_replaces_managed_files_only(tmp_path):
    public = tmp_path / 'public'
    unknown = result(public, 'unknown')
    old = tmp_path / 'old'
    new = tmp_path / 'new'
    result(old, 'old-model')
    result(new, 'new-model')
    runner._publish_resume_results(old, public)
    runner._publish_resume_results(new, public)
    assert not (public / 'results/old-model/master/task.json').exists()
    assert (public / 'results/new-model/master/task.json').is_file()
    assert unknown.is_file()
    # Missing/empty new results must not keep managed stale results alive.
    runner._publish_resume_results(tmp_path / 'empty', public)
    assert not (public / 'results/new-model/master/task.json').exists()
    assert unknown.is_file()


def test_run_publication_keeps_current_model_union(tmp_path, monkeypatch):
    args = MTEBEvalConfig(output_folder=str(tmp_path), task_names=['task'], offline=False,
                          allow_download=True)
    models = [MTEBModelConfig(api_base='https://example.invalid/v1', model_name=name) for name in ['one', 'two']]
    old = tmp_path / '.rag_resume/old'
    result(old, 'old')
    runner._publish_resume_results(old, tmp_path)
    unknown = result(tmp_path, 'unknown')
    monkeypatch.setattr(runner, '_offline_dataset_guard', lambda enabled: nullcontext())

    def evaluate(model, config):
        assert not (tmp_path / 'results/old/master/task.json').exists()
        checkpoint = tmp_path / '.rag_resume' / model.model_name
        result(checkpoint, model.model_name)
        runner._publish_resume_results(checkpoint, config.output_folder)
        return model.model_name

    monkeypatch.setattr(runner, 'one_stage_eval', evaluate)
    assert runner.run_mteb_eval(SimpleNamespace(eval=args, models=models)) == 'two'
    assert (tmp_path / 'results/one/master/task.json').is_file()
    assert (tmp_path / 'results/two/master/task.json').is_file()
    assert unknown.is_file()
    # Next run must replace the previous run's union, not keep model two.
    runner.run_mteb_eval(SimpleNamespace(eval=args, models=models[:1]))
    assert not (tmp_path / 'results/two/master/task.json').exists()


@pytest.mark.parametrize('predictions_present', [False, True])
def test_stage_one_recomputes_predictions_and_publishes_original_path(
        tmp_path, monkeypatch, predictions_present):
    args = MTEBEvalConfig(output_folder=str(tmp_path), task_names=['task'])
    encoder_args = MTEBModelConfig(api_base='https://example.invalid/v1', model_name='encoder')
    reranker_args = MTEBModelConfig(api_base='https://example.invalid/v1', model_name='reranker', is_cross_encoder=True)
    encoder, reranker = SimpleNamespace(), SimpleNamespace()
    checkpoint = tmp_path / '.rag_resume/encoder'
    result(checkpoint, 'encoder')  # Completed metrics must not skip missing predictions.
    requests = checkpoint / 'requests/cached-request'
    requests.parent.mkdir(parents=True)
    requests.write_text('cached response')
    predictions = checkpoint / 'predictions'
    if predictions_present:
        predictions.mkdir()
        (predictions / 'partial.json').write_text('{}')
    monkeypatch.setattr(runner, 'load_model',
                        lambda config: encoder if config is encoder_args else reranker)

    def configure(model, model_args, eval_args):
        model.request_cache_dir = str(checkpoint / 'requests')
        return str(checkpoint)

    monkeypatch.setattr(runner, '_configure_resume', configure)
    conversions = []
    task = SimpleNamespace(metadata=SimpleNamespace(type='Retrieval'), load_data=lambda: None,
                           convert_to_reranking=lambda **kwargs: conversions.append(kwargs))
    monkeypatch.setattr(runner, 'resolve_tasks', lambda config: [task])
    monkeypatch.setattr(runner, '_preflight_offline_tasks', lambda *args: None)
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        assert kwargs['overwrite_strategy'] == 'always'
        assert Path(encoder.request_cache_dir, 'cached-request').read_text() == 'cached response'
        assert kwargs['prediction_folder'] == predictions
        predictions.mkdir(exist_ok=True)
        (predictions / 'complete.json').write_text('{}')
        return SimpleNamespace(task_results=[])

    monkeypatch.setattr(runner.mteb, 'evaluate', evaluate)
    runner.two_stage_eval(encoder_args, reranker_args, args)
    assert len(calls) == 1  # Implementing stage two is explicitly outside this fix.
    assert conversions == [{'top_ranked_path': str(predictions), 'top_k': args.top_k}]
    assert (tmp_path / 'stage1/results/encoder/master/task.json').is_file()
    assert not (tmp_path / 'results/encoder/master/task.json').exists()
