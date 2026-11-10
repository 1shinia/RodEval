"""RAG-only resume wiring regressions (offline, no provider calls)."""
from types import SimpleNamespace
from pathlib import Path

from evalscope.backend.rag_eval.mteb import task_template as runner
from evalscope.backend.rag_eval.mteb.arguments import MTEBModelConfig, MTEBEvalConfig


def test_completed_results_are_reused_only_in_same_config(tmp_path):
    model = MTEBModelConfig(model_name='audit', api_base='https://example.invalid/v1', api_key='first')
    args = MTEBEvalConfig(output_folder=str(tmp_path), task_names=['audit'], limits=2)
    loaded = SimpleNamespace()
    cache_dir = runner._configure_resume(loaded, model, args)
    assert Path(loaded.request_cache_dir).is_relative_to(tmp_path)
    first_namespace = loaded.request_cache_namespace
    assert 'first' not in first_namespace
    kwargs = runner._build_evaluate_kwargs(args, cache_dir)
    assert kwargs['overwrite_strategy'] == 'only-missing'
    model.api_key = 'replacement'
    assert runner._configure_resume(SimpleNamespace(), model, args) == cache_dir
    args.limits = 3
    other = SimpleNamespace()
    assert runner._configure_resume(other, model, args) != cache_dir
    assert other.request_cache_namespace != first_namespace


def test_noncredential_tokenizer_parameters_change_cache_identity(tmp_path):
    args = MTEBEvalConfig(output_folder=str(tmp_path), task_names=['audit'])
    model = MTEBModelConfig(model_name='audit', api_base='https://example.invalid/v1',
                            model_kwargs={'max_tokens': 10, 'tokenizer_name': 'one'})
    first = runner._configure_resume(SimpleNamespace(), model, args)
    model.model_kwargs['max_tokens'] = 20
    assert runner._configure_resume(SimpleNamespace(), model, args) != first
    model.model_kwargs['max_tokens'] = 10
    model.model_kwargs['tokenizer_name'] = 'two'
    assert runner._configure_resume(SimpleNamespace(), model, args) != first


def test_local_model_weight_change_invalidates_checkpoint(tmp_path):
    folder = tmp_path / 'model'
    folder.mkdir()
    weights = folder / 'model.safetensors'
    weights.write_bytes(b'first')
    args = MTEBEvalConfig(output_folder=str(tmp_path / 'run'), task_names=['audit'])
    model = MTEBModelConfig(model_name_or_path=str(folder))
    first = runner._configure_resume(SimpleNamespace(), model, args)
    weights.write_bytes(b'changed-weights')
    assert runner._configure_resume(SimpleNamespace(), model, args) != first


def test_one_stage_wires_resume_and_publishes_results(tmp_path, monkeypatch):
    loaded = SimpleNamespace()
    model = MTEBModelConfig(model_name='audit', api_base='https://example.invalid/v1')
    args = MTEBEvalConfig(output_folder=str(tmp_path), task_names=['audit'], limits=1)
    task = SimpleNamespace(metadata=SimpleNamespace(name='audit', type='STS'),
                           dataset=None, load_data=lambda: None)
    monkeypatch.setattr(runner, 'load_model', lambda config: loaded)
    monkeypatch.setattr(runner, 'resolve_tasks', lambda config: [task])
    monkeypatch.setattr(runner, '_preflight_offline_tasks', lambda *a: None)
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        assert loaded.request_cache_dir
        # Exercise the real ResultCache layout used by report parsing.
        result_dir = Path(kwargs['cache'].cache_path) / 'results' / 'audit' / 'master'
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / 'audit.json').write_text('{"task_name":"audit","scores":{}}')
        return SimpleNamespace(task_results=[])
    monkeypatch.setattr(runner.mteb, 'evaluate', evaluate)
    runner.one_stage_eval(model, args)
    assert calls[0]['overwrite_strategy'] == 'only-missing'
    assert (tmp_path / 'results' / 'audit' / 'master' / 'audit.json').is_file()
    assert (tmp_path / 'progress.json').read_text().find('completed') >= 0
