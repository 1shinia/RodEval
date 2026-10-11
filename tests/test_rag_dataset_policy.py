import copy

from evalscope.service.blueprints.eval import _apply_rag_dataset_policy


def test_admin_keeps_mteb_download_configuration():
    config = {'tool': 'mteb', 'eval': {'hub': 'huggingface'}}

    result = _apply_rag_dataset_policy(config, is_admin=True)

    assert result == config
    assert result is not config


def test_normal_user_is_forced_offline_even_if_hub_is_forged():
    config = {'tool': 'mteb', 'eval': {'hub': 'huggingface'}}
    original = copy.deepcopy(config)

    result = _apply_rag_dataset_policy(config, is_admin=False)

    assert config == original
    assert result['eval']['offline'] is True
    assert result['eval']['allow_download'] is False
    assert result['eval']['hub'] == 'huggingface'


def test_non_mteb_rag_tools_are_not_changed():
    config = {'tool': 'ragas', 'eval': {'testset_file': '/data/testset.json'}}

    result = _apply_rag_dataset_policy(config, is_admin=False)

    assert result == config
    assert result is not config


def test_rag_offline_preflight_reports_missing_cache_before_evaluation():
    from evalscope.backend.rag_eval.mteb.task_template import _preflight_offline_tasks

    class Metadata:
        name = 'STSBenchmark'

    class MissingCacheTask:
        metadata = Metadata()
        data_loaded = False

        def load_data(self):
            raise FileNotFoundError('cached dataset files are missing')

    try:
        _preflight_offline_tasks([MissingCacheTask()], offline=True, allow_download=False)
    except RuntimeError as exc:
        assert 'RAG 数据集缓存不可用' in str(exc)
        assert 'STSBenchmark' in str(exc)
    else:
        raise AssertionError('missing RAG cache must fail before evaluation')


def test_rag_offline_guard_enables_and_restores_runtime_flags(monkeypatch):
    from evalscope.backend.rag_eval.mteb.task_template import _offline_dataset_guard

    from datasets import config as datasets_config
    from huggingface_hub import constants as hf_constants

    monkeypatch.delenv('HF_DATASETS_OFFLINE', raising=False)
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    monkeypatch.setattr(datasets_config, 'HF_HUB_OFFLINE', False)
    monkeypatch.setattr(datasets_config, 'HF_DATASETS_OFFLINE', False)
    monkeypatch.setattr(hf_constants, 'HF_HUB_OFFLINE', False)

    with _offline_dataset_guard(True):
        import os

        assert os.environ['HF_DATASETS_OFFLINE'] == '1'
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
        assert datasets_config.HF_HUB_OFFLINE is True
        assert datasets_config.HF_DATASETS_OFFLINE is True
        assert hf_constants.HF_HUB_OFFLINE is True

    assert datasets_config.HF_HUB_OFFLINE is False
    assert datasets_config.HF_DATASETS_OFFLINE is False
    assert hf_constants.HF_HUB_OFFLINE is False
