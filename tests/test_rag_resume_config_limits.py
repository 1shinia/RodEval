"""The display-count rewrite must not invalidate full-run checkpoints."""
from types import SimpleNamespace
import yaml
from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig, MTEBModelConfig
from evalscope.backend.rag_eval.mteb.task_template import _configure_resume, _update_config_limits


def test_full_run_resume_preserves_original_limits_identity(tmp_path):
    args = MTEBEvalConfig(output_folder=str(tmp_path), task_names=['audit'])
    model = MTEBModelConfig(model_name='audit', api_base='https://example.invalid/v1')
    first = _configure_resume(SimpleNamespace(), model, args)
    folder = tmp_path / 'configs'
    folder.mkdir()
    path = folder / 'task_config.yaml'
    path.write_text(yaml.safe_dump({'eval_config': {'eval': args.model_dump()}}))
    _update_config_limits(str(tmp_path), 20)
    saved = yaml.safe_load(path.read_text())['eval_config']['eval']
    assert saved['limits'] == 20  # reports retain the existing actual-count behavior
    resumed = MTEBEvalConfig(**saved)
    assert _configure_resume(SimpleNamespace(), model, resumed) == first
