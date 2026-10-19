from unittest.mock import Mock

from evalscope.api.dataset.loader import RemoteDataLoader


def test_offline_loader_never_deletes_cache_for_force_redownload(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    from datasets import Dataset
    import evalscope.api.dataset.loader as module
    loader = RemoteDataLoader(data_id_or_path='audit/data', split='test', dataset_dir=str(tmp_path), force_redownload=True)
    monkeypatch.setattr(module, 'gen_hash', lambda value: 'audit')
    monkeypatch.setattr(module, 'safe_filename', lambda value: 'data')
    cache = tmp_path / 'datasets' / 'data-audit'
    Dataset.from_dict({'question': ['cached']}).save_to_disk(str(cache))
    remove = Mock(side_effect=AssertionError('offline must not delete cache'))
    monkeypatch.setattr(module.shutil, 'rmtree', remove)
    from evalscope.api.dataset import Sample
    monkeypatch.setattr(module, 'record_to_sample_fn', lambda *a: lambda record: Sample(input=record['question']))
    result = loader.load()
    remove.assert_not_called()
    assert cache.is_dir()
    assert result[0].input == 'cached'
