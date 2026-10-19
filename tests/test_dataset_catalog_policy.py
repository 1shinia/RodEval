"""Dataset-level publication authorization; no model workers run."""
import pytest
from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, EvalBackend
from evalscope.service.blueprints import eval as routes
from evalscope.service import dataset_catalog_inventory as inventory


@pytest.fixture
def approved(monkeypatch):
    monkeypatch.setattr('evalscope.service.utils.benchmarks.load_benchmark_data',
                        lambda name: {name: {'meta': {'category': 'llm', 'few_shot_num': 3}}})
    monkeypatch.setattr(inventory, 'require_dataset_published', lambda *args: {})


def test_unpublished_text_is_rejected(approved, monkeypatch):
    def deny(*args):
        raise ValueError('数据集尚未发布，请联系管理员')
    monkeypatch.setattr(inventory, 'require_dataset_published', deny)
    with pytest.raises(routes.RequestValidationError, match='管理员'):
        routes._apply_eval_dataset_policy({'datasets': ['arc']}, False)


def test_explicit_subsets_and_sampling_preserved(approved):
    args = {'subset_list': ['Easy', 'Challenge'], 'few_shot_num': 4, 'shuffle': True,
            '_catalog_selection': [{'id': 1, 'fingerprint': 'old'}]}
    data = {'datasets': ['arc'], 'dataset_args': {'arc': args}}
    result = routes._apply_eval_dataset_policy(data, False)
    assert result['dataset_args']['arc'] == {
        'subset_list': ['Easy', 'Challenge'], 'few_shot_num': 4, 'shuffle': True}
    assert result['dataset_dir'] == DEFAULT_DATASET_CACHE_DIR
    assert result['dataset_offline'] is True
    assert '_catalog_selection' in data['dataset_args']['arc']
    assert routes._apply_eval_dataset_policy(result, False) == result


def test_old_fingerprint_does_not_block_approved_dataset(approved):
    result = routes._apply_eval_dataset_policy({'datasets': ['arc'], 'dataset_args': {
        'arc': {'_catalog_selection': [{'id': 1, 'fingerprint': 'old'}]}}}, False)
    assert '_catalog_selection' not in result['dataset_args']['arc']


def test_normal_user_cannot_override_cache_source(approved):
    result = routes._apply_eval_dataset_policy({'datasets': ['arc'],
        'dataset_dir': '/other', 'dataset_hub': 'huggingface', 'dataset_args': {
            'arc': {'dataset_id': 'other', 'local_path': '/other', 'force_redownload': True}}}, False)
    assert result['dataset_dir'] == DEFAULT_DATASET_CACHE_DIR
    assert result['dataset_hub'] == 'modelscope'
    assert result['dataset_args']['arc'] == {}


@pytest.mark.parametrize('published', [True, False])
def test_agent_uses_publication_and_offline_policy(monkeypatch, published):
    monkeypatch.setattr('evalscope.service.utils.benchmarks.load_benchmark_data',
                        lambda name: {name: {'meta': {'category': 'agent'}}})
    checked = []

    def authorize(category, name):
        checked.append((category, name))
        if not published:
            raise inventory.catalog.CatalogError('数据集尚未发布。', 403)
        return {}

    monkeypatch.setattr(inventory, 'require_dataset_published', authorize)
    data = {'datasets': ['bfcl_v3'], 'dataset_hub': 'huggingface',
            'dataset_dir': '/other', 'dataset_args': {'bfcl_v3': {
                'dataset_id': 'other', 'local_path': '/other', 'force_redownload': True,
                'subset_list': ['simple'], 'few_shot_num': 0}}}
    if not published:
        with pytest.raises(routes.RequestValidationError, match='尚未发布'):
            routes._apply_eval_dataset_policy(data, False)
    else:
        result = routes._apply_eval_dataset_policy(data, False)
        assert result['dataset_offline'] is True
        assert result['dataset_dir'] == DEFAULT_DATASET_CACHE_DIR
        assert result['dataset_hub'] == 'modelscope'
        assert result['dataset_args']['bfcl_v3'] == {
            'subset_list': ['simple'], 'few_shot_num': 0}
    assert checked == [('agent', 'bfcl_v3')]


def test_admin_keeps_existing_behavior():
    data = {'datasets': ['arc'], 'dataset_hub': 'huggingface'}
    assert routes._apply_eval_dataset_policy(data, True) == data


@pytest.mark.parametrize('backend', [EvalBackend.AIGC_EVAL, EvalBackend.AUDIO_EVAL])
def test_other_modules_not_subject_to_publication(backend):
    data = {'eval_backend': backend, 'eval_config': {'untouched': True}}
    assert routes._apply_eval_dataset_policy(data, False) == data


def test_rag_cannot_expand_by_type():
    with pytest.raises(routes.RequestValidationError, match='RAG'):
        routes._apply_eval_dataset_policy({'eval_backend': EvalBackend.RAG_EVAL,
            'eval_config': {'tool': 'mteb', 'eval': {'task_types': ['Retrieval']}}}, False)
