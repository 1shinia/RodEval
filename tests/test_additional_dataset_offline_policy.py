"""Dataset-only offline guards; all hub calls are mocked (no downloads)."""
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

BASE = 'evalscope.benchmarks.'
CASES = [
    ('data_collection.data_collection_adapter', 'DataCollectionAdapter', 'test.jsonl'),
    ('healthbench.healthbench_adapter', 'HealthBenchAdapter', 'health.jsonl'),
    ('needle_haystack.needle_haystack_adapter', 'NeedleHaystackAdapter', 'PaulGraham_Essays.txt'),
    ('tir_bench.tir_bench_adapter', 'TIRBenchAdapter', 'data.zip'),
]


@pytest.fixture(autouse=True)
def local_snapshot_api(monkeypatch):
    # SDK 1.38 dataset_snapshot_download lacks local_files_only; snapshot_download supports repo_type.
    def local_snapshot(*args, **kwargs):
        import modelscope
        assert kwargs.pop('repo_type') == 'dataset'
        assert kwargs.get('local_files_only') is True
        return modelscope.dataset_snapshot_download(*args, **kwargs)
    monkeypatch.setattr(importlib.import_module('modelscope.hub.snapshot_download'), 'snapshot_download', local_snapshot)


def adapter(module, cls):
    obj = SimpleNamespace()
    obj.record_to_sample = Mock()
    obj.load = lambda: getattr(module, cls).load(obj)
    obj.dataset_id = 'example/dataset'
    obj.eval_split = 'test'
    obj.version_file = 'health.jsonl'
    obj.subset_list = ['english']
    obj.default_subset = 'default'
    obj.limit = None
    obj.repeats = 1
    obj.shuffle = False
    obj._init_tokenizer = Mock()
    obj._init_length = Mock()
    obj.context_lengths = []
    obj.document_depth_percents = []
    obj._get_context_tokens = Mock(return_value=[])
    obj.load_from_remote = Mock(return_value=({}, None))
    return obj


@pytest.mark.parametrize('mod,cls,required', CASES)
def test_offline_missing_snapshot_never_online(monkeypatch, tmp_path, mod, cls, required):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    module = importlib.import_module(BASE + mod)
    obj = adapter(module, cls)

    def snapshot(*args, **kwargs):
        assert kwargs.get('local_files_only') is True, 'online snapshot forbidden'
        return str(tmp_path)

    monkeypatch.setattr('modelscope.dataset_snapshot_download', snapshot)
    monkeypatch.setattr('evalscope.api.dataset.hub.download_dataset_file',
                        Mock(side_effect=RuntimeError('请联系管理员准备数据集')))
    with pytest.raises(RuntimeError, match='管理员'):
        obj.load()


@pytest.mark.parametrize('mod,cls,required', CASES[:3])
def test_offline_complete_snapshot(monkeypatch, tmp_path, mod, cls, required):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', 'true')
    module = importlib.import_module(BASE + mod)
    obj = adapter(module, cls)
    (tmp_path / required).write_text('cached corpus' if required.endswith('.txt') else '{}\n')
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', snapshot)
    monkeypatch.setattr('evalscope.api.dataset.hub.download_dataset_file',
                        Mock(return_value=str(tmp_path / required)))
    if hasattr(module, 'LocalDataLoader'):
        monkeypatch.setattr(module, 'LocalDataLoader', Mock(return_value=SimpleNamespace(load=lambda: [])))
    if cls == 'HealthBenchAdapter':
        monkeypatch.setattr(module, 'DatasetDict', SimpleNamespace(from_dataset=lambda **kw: {}))
    obj.load()
    if snapshot.called:
        assert snapshot.call_args.kwargs.get('local_files_only') is True


@pytest.mark.parametrize('mod,cls,required', CASES)
def test_online_snapshot_behavior(monkeypatch, tmp_path, mod, cls, required):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    module = importlib.import_module(BASE + mod)
    obj = adapter(module, cls)
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', snapshot)
    if hasattr(module, 'LocalDataLoader'):
        monkeypatch.setattr(module, 'LocalDataLoader', Mock(return_value=SimpleNamespace(load=lambda: [])))
    if cls == 'HealthBenchAdapter':
        monkeypatch.setattr(module, 'DatasetDict', SimpleNamespace(from_dataset=lambda **kw: {}))
    obj.load()
    assert snapshot.call_count == 1
    assert not snapshot.call_args.kwargs.get('local_files_only', False)


@pytest.mark.parametrize('extension', ['csv', 'tsv'])
def test_collection_local_supported_formats(monkeypatch, tmp_path, extension):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    module = importlib.import_module(BASE + CASES[0][0])
    obj = adapter(module, CASES[0][1])
    obj.dataset_id = str(tmp_path)
    (tmp_path / f'test.{extension}').write_text('prompt\nexample\n')
    monkeypatch.setattr(module, 'LocalDataLoader', Mock(return_value=SimpleNamespace(load=lambda: [])))
    obj.load()


@pytest.mark.parametrize('extracted', [False, True])
def test_tir_complete_images_cache(monkeypatch, tmp_path, extracted):
    import zipfile
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    module = importlib.import_module(BASE + CASES[3][0])
    obj = adapter(module, CASES[3][1])
    if extracted:
        (tmp_path / 'data').mkdir()
        (tmp_path / 'data' / 'image.png').write_bytes(b'image')
    else:
        with zipfile.ZipFile(tmp_path / 'data.zip', 'w') as archive:
            archive.writestr('data/image.png', b'image')
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', snapshot)
    obj.load()
    assert (tmp_path / 'data' / 'image.png').is_file()
    assert snapshot.call_args.kwargs['local_files_only'] is True
    obj.load_from_remote.assert_called_once()


def test_tir_missing_referenced_image(monkeypatch, tmp_path):
    module = importlib.import_module(BASE + CASES[3][0])
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    obj = adapter(module, CASES[3][1])
    obj.image_root = str(tmp_path)
    with pytest.raises(RuntimeError, match='管理员'):
        module.TIRBenchAdapter.record_to_sample(obj, {'image_1': 'data/missing.png'})


def test_air_chat_all_metadata_audio_required(monkeypatch, tmp_path):
    from evalscope.benchmarks.air_bench.utils import download_air_bench
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    root, folder, records = air_cache(tmp_path, 'Chat', 'speech_QA', 'iemocap')
    (root / 'Chat' / 'Chat_meta.json').write_text(json.dumps(records))
    with pytest.raises(RuntimeError, match='管理员'):
        download_air_bench('Chat', 'example/dataset', str(tmp_path))
    (root / 'Chat' / folder / 'clip.wav').write_bytes(b'audio')
    assert download_air_bench('Chat', 'example/dataset', str(tmp_path)) == str(root)


def test_air_online_preserves_download(monkeypatch, tmp_path):
    from evalscope.benchmarks.air_bench.utils import download_air_bench
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    remote = Mock(return_value=str(tmp_path))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', remote)
    assert download_air_bench('Chat', 'example/dataset', str(tmp_path), ['speech_QA_iemocap']) == str(tmp_path)
    assert remote.call_args.kwargs['allow_file_pattern'] == ['Chat/Chat_meta.json', 'Chat/speech_QA_iemocap/*']
    assert not remote.call_args.kwargs.get('local_files_only', False)


def air_cache(tmp_path, track='Foundation', task='Audio_Grounding', dataset='AudioGrounding'):
    folder = f'{task}_{dataset}'
    root = tmp_path / 'example' / 'dataset'
    (root / track / folder).mkdir(parents=True)
    records = [{'task_name': task, 'dataset_name': dataset, 'path': 'clip.wav'}]
    return root, folder, records


@pytest.mark.parametrize('state', ['no_meta', 'no_audio', 'wrong_extension', 'complete'])
@pytest.mark.parametrize('local', [False, True])
def test_air_requires_meta_and_real_audio(monkeypatch, tmp_path, state, local):
    from evalscope.benchmarks.air_bench.utils import download_air_bench
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', 'on')
    root, folder, records = air_cache(tmp_path)
    if state != 'no_meta':
        (root / 'Foundation' / 'Foundation_meta.json').write_text(json.dumps(records))
    if state in ('wrong_extension', 'complete'):
        suffix = 'flac' if state == 'complete' else 'wav'
        (root / 'Foundation' / folder / f'clip.{suffix}').write_bytes(b'audio')
    remote = Mock(side_effect=AssertionError('network download forbidden'))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', remote)
    kwargs: dict = dict(track='Foundation', dataset_id=str(root) if local else 'example/dataset',
                  cache_dir=str(tmp_path), subset_dirs=[folder])
    if state == 'complete':
        assert download_air_bench(**kwargs) == str(root)
    else:
        with pytest.raises(RuntimeError, match='管理员'):
            download_air_bench(**kwargs)
    remote.assert_not_called()


def test_air_cold_offline_no_snapshot(monkeypatch, tmp_path):
    from evalscope.benchmarks.air_bench.utils import download_air_bench
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', 'yes')
    remote = Mock(side_effect=AssertionError('network download forbidden'))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', remote)
    with pytest.raises(RuntimeError, match='管理员'):
        download_air_bench('Chat', 'example/dataset', str(tmp_path), ['speech_QA_iemocap'])
    remote.assert_not_called()
