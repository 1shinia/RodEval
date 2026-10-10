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
