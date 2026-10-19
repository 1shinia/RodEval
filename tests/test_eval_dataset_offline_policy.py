import copy
import os

import pytest

from evalscope.api.dataset.hub import download_dataset_file
from evalscope.api.dataset.loader import RemoteDataLoader
from evalscope.config import TaskConfig
from evalscope.constants import DEFAULT_DATASET_CACHE_DIR
from evalscope.service.blueprints.eval import _apply_eval_dataset_policy
from evalscope.service.utils import process as process_utils


def test_admin_keeps_llm_vlm_dataset_configuration():
    data = {
        'datasets': ['gsm8k'],
        'dataset_hub': 'huggingface',
        'dataset_dir': '/admin/cache',
        'dataset_args': {'gsm8k': {'local_path': '/admin/data', 'subset_list': ['main']}},
    }

    result = _apply_eval_dataset_policy(data, is_admin=True)

    assert result == data
    assert result is not data


@pytest.fixture
def published(monkeypatch):
    from evalscope.service import dataset_catalog_inventory
    monkeypatch.setattr(dataset_catalog_inventory, 'require_dataset_published', lambda *a: dict(
        benchmark='gsm8k', category='llm', dataset_id='modelscope/gsm8k', revision=None,
        subset='main', split='test', dataset_hub='modelscope', dataset_dir=DEFAULT_DATASET_CACHE_DIR,
        details={'few_shot_num': 4}))
    return {'_catalog_selection': [{'id': 1, 'fingerprint': 'a' * 64}]}


def test_normal_user_is_forced_to_managed_offline_cache(published):
    data = {
        'datasets': ['gsm8k'],
        'dataset_hub': 'huggingface',
        'dataset_dir': '/forged/cache',
        'dataset_args': {
            'gsm8k': {
                **published,
                'local_path': '/forged/data',
                'force_redownload': True,
                'subset_list': ['main'],
            }
        },
    }
    original = copy.deepcopy(data)

    result = _apply_eval_dataset_policy(data, is_admin=False)

    assert data == original
    assert result['dataset_offline'] is True
    assert result['dataset_hub'] == 'modelscope'
    assert result['dataset_dir'] == DEFAULT_DATASET_CACHE_DIR
    args = result['dataset_args']['gsm8k']
    assert 'local_path' not in args
    assert not args.get('force_redownload', False)
    assert '_catalog_selection' not in args
    assert args['subset_list'] == ['main']


def test_normal_user_offline_policy_survives_task_config_roundtrip(published):
    normalized = _apply_eval_dataset_policy(
        {'model': 'demo', 'datasets': ['gsm8k'], 'dataset_args': {'gsm8k': published},
         'api_url': 'http://example.test/v1'},
        is_admin=False,
    )

    config = TaskConfig.from_dict(normalized)

    assert config.dataset_offline is True
    assert config.to_dict()['dataset_offline'] is True


def test_offline_remote_loader_rejects_missing_processed_cache_before_hub_access(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    loader = RemoteDataLoader(
        data_id_or_path='missing-dataset',
        split='test',
        dataset_dir=str(tmp_path),
    )

    def unexpected_hub_access(*args, **kwargs):
        raise AssertionError('offline loading must not access a dataset hub')

    monkeypatch.setattr('evalscope.api.dataset.loader.DatasetHub', unexpected_hub_access)

    with pytest.raises(RuntimeError, match='尚未由管理员准备'):
        loader.load()


def test_offline_direct_file_download_is_blocked_before_sdk_access(monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')

    with pytest.raises(RuntimeError, match='尚未由管理员准备'):
        download_dataset_file('missing-dataset', 'data/test.jsonl', data_source='modelscope')


def test_eval_worker_sets_offline_environment_only_inside_child(monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    monkeypatch.delenv('HF_DATASETS_OFFLINE', raising=False)
    seen = {}

    def fake_run_task(config):
        seen.update({
            key: os.environ.get(key)
            for key in (
                'EVALSCOPE_DATASET_OFFLINE',
                'HF_DATASETS_OFFLINE',
                'HF_HUB_OFFLINE',
                'TRANSFORMERS_OFFLINE',
            )
        })
        return 'ok'

    monkeypatch.setattr(process_utils, 'run_task', fake_run_task)
    config = TaskConfig(model='demo', datasets=['gsm8k'], dataset_offline=True)

    assert process_utils.run_eval_wrapper(config) == 'ok'
    assert seen == {
        'EVALSCOPE_DATASET_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    }
    assert os.environ.get('EVALSCOPE_DATASET_OFFLINE') is None
