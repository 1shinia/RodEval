"""Dataset-level publication authorization, independent of cache snapshots."""
from copy import deepcopy

import pytest

from evalscope.constants import EvalBackend
from evalscope.service import dataset_catalog_inventory as inventory
from evalscope.service import dataset_catalog_policy as policy
from evalscope.service.blueprints import eval as routes


@pytest.fixture
def published(monkeypatch):
    calls = []

    def require(category, name):
        calls.append((category, name))
        if name == 'private':
            raise ValueError('数据集未发布，请联系管理员')
        # No cache version, fingerprint or source is required for publication.
        return {'category': category, 'benchmark': name, 'published': True}

    monkeypatch.setattr(inventory, 'require_dataset_published', require, raising=False)
    monkeypatch.setattr('evalscope.service.utils.benchmarks.load_benchmark_data',
                        lambda name: {name: {'meta': {'category': {'arc': 'llm', 'vision': 'vlm',
                            'private': 'llm'}.get(name)}}})
    return calls


def test_published_dataset_preserves_user_sampling_without_version(published):
    args = {'subset_list': ['chosen'], 'eval_split': 'validation', 'train_split': 'train2',
            'few_shot_num': 7, 'shuffle': True, 'filters': {'keep': ['a']}}
    data = {'datasets': ['arc', 'vision'], 'dataset_args': {'arc': deepcopy(args)}}
    result = policy.apply_catalog_policy(data)
    assert result['dataset_args']['arc'] == args
    assert published == [('llm', 'arc'), ('vlm', 'vision')]


def test_old_selections_removed_even_when_stale(published):
    result = policy.apply_catalog_policy({'datasets': ['arc'], 'dataset_args': {
        'arc': {'_catalog_selection': [{'id': 123, 'fingerprint': 'stale'}]},
        'unused': {'_catalog_selection': ['stale'], 'few_shot_num': 2}}})
    assert result['dataset_args'] == {'arc': {}, 'unused': {'few_shot_num': 2}}


@pytest.mark.parametrize('name', ['unknown', 'private'])
def test_unknown_or_unpublished_rejected(name, published):
    with pytest.raises(ValueError):
        policy.apply_catalog_policy({'datasets': [name]})


def test_unpublished_second_dataset_is_checked(published):
    with pytest.raises(ValueError, match='未发布'):
        policy.apply_catalog_policy({'datasets': ['arc', 'private']})
    assert published == [('llm', 'arc'), ('llm', 'private')]


@pytest.mark.parametrize('names', [None, [], 'STS', [''], [None], [['STS']]])
def test_rag_requires_explicit_valid_names(names, published):
    with pytest.raises(ValueError):
        policy.apply_rag_catalog_policy({'eval': {'task_names': names, 'task_types': ['STS']}})


def test_rag_authorizes_each_name_and_removes_snapshot_selection(published):
    config = {'eval': {'task_names': ['STS', 'Classification', 'Retrieval'],
                      'catalog_selection': [{'id': 1, 'fingerprint': 'old'}],
                      'task_types': ['Clustering'], 'languages': ['en'],
                      'limits': 5, 'offline': False, 'allow_download': True}}
    result = policy.apply_rag_catalog_policy(config)['eval']
    assert published == [('rag', name) for name in ['STS', 'Classification', 'Retrieval']]
    assert result == {'task_names': ['STS', 'Classification', 'Retrieval'],
                      'limits': 5, 'offline': True, 'allow_download': False}


def test_rag_unpublished_rejected(published):
    with pytest.raises(ValueError, match='未发布'):
        policy.apply_rag_catalog_policy({'eval': {'task_names': ['STS', 'private']}})


def test_rag_custom_paths_rejected(published):
    with pytest.raises(ValueError, match='自定义'):
        policy.apply_rag_catalog_policy({'eval': {'task_names': ['STS'],
            'custom_tasks': [{'name': 'evil', 'data_path': '/tmp/data'}]}})


def test_route_keeps_recursive_source_stripping_and_forces_offline(published):
    data = {'datasets': ['arc'], 'dataset_offline': False, 'dataset_dir': '/evil',
            'dataset_hub': 'huggingface', 'dataset_args': {'arc': {
                'dataset_id': '/evil', 'subset_list': ['chosen'], 'few_shot_num': 7,
                '_catalog_selection': [{'id': 1}],
                'nested': [{'cache_dir': '/evil', 'data_files': '/evil', 'keep': True}]}}}
    original = deepcopy(data)
    result = routes._apply_eval_dataset_policy(data, False)
    assert data == original
    assert result['dataset_offline'] is True
    assert result['dataset_dir'] != '/evil'
    assert result['dataset_hub'] != 'huggingface'
    assert result['dataset_args']['arc'] == {
        'subset_list': ['chosen'], 'few_shot_num': 7, 'nested': [{'keep': True}]}


def test_admin_bypasses_publication_and_keeps_snapshot_selection(published):
    data = {'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': {
        'tool': 'mteb', 'eval': {'catalog_selection': [{'id': 1}], 'offline': False}}}
    assert routes._apply_eval_dataset_policy(data, True) == data
    assert not published


@pytest.mark.parametrize('args', [[], {'arc': []}])
def test_invalid_dataset_arguments_rejected(args, published):
    with pytest.raises(ValueError):
        policy.apply_catalog_policy({'datasets': ['arc'], 'dataset_args': args})
