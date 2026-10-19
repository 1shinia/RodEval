"""Real SDK cache misses must fail before any socket connection."""
import socket
from unittest.mock import Mock

import pytest

from evalscope.api.dataset.hub import download_dataset_file


@pytest.mark.parametrize('source', ['modelscope', 'huggingface'])
def test_real_sdk_offline_cache_miss_never_connects(source, tmp_path, monkeypatch):
    # Missing dependency must fail, not masquerade as a successful offline denial.
    if source == 'modelscope':
        from modelscope.hub.file_download import dataset_file_download
        assert callable(dataset_file_download)
    else:
        from huggingface_hub import hf_hub_download
        assert callable(hf_hub_download)
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    connect = Mock(side_effect=AssertionError('unexpected network access'))
    monkeypatch.setattr(socket.socket, 'connect', connect)
    with pytest.raises(RuntimeError, match='管理员'):
        download_dataset_file('audit-missing/dataset', 'missing.zip', source, cache_dir=str(tmp_path))
    connect.assert_not_called()


def test_real_modelscope_legacy_cache_hit_never_connects(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    target = tmp_path / 'audit' / 'dataset' / 'media.zip'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'cached')
    connect = Mock(side_effect=AssertionError('unexpected network access'))
    monkeypatch.setattr(socket.socket, 'connect', connect)
    result = download_dataset_file('audit/dataset', 'media.zip', 'modelscope', cache_dir=str(tmp_path))
    assert result == str(target)
    assert target.read_bytes() == b'cached'
    connect.assert_not_called()


def test_real_modelscope_snapshot_local_only(tmp_path, monkeypatch):
    from modelscope.hub.snapshot_download import snapshot_download

    target = tmp_path / 'audit' / 'dataset' / 'test.jsonl'
    target.parent.mkdir(parents=True)
    target.write_text('{}\n')
    connect = Mock(side_effect=AssertionError('unexpected network access'))
    monkeypatch.setattr(socket.socket, 'connect', connect)
    resolved = snapshot_download(
        'audit/dataset', repo_type='dataset', cache_dir=str(tmp_path),
        allow_file_pattern='*.jsonl', local_files_only=True,
    )
    assert (target.parent / 'test.jsonl').read_text() == '{}\n'
    assert str(target.parent) == resolved
    with pytest.raises(Exception):
        snapshot_download('audit/missing', repo_type='dataset', cache_dir=str(tmp_path), local_files_only=True)
    connect.assert_not_called()


def test_real_huggingface_cache_hit_never_connects(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    repo = tmp_path / 'datasets--audit--dataset'
    commit = 'a' * 40
    (repo / 'refs').mkdir(parents=True)
    (repo / 'refs' / 'main').write_text(commit)
    target = repo / 'snapshots' / commit / 'media.zip'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'cached')
    connect = Mock(side_effect=AssertionError('unexpected network access'))
    monkeypatch.setattr(socket.socket, 'connect', connect)
    result = download_dataset_file('audit/dataset', 'media.zip', 'huggingface', cache_dir=str(tmp_path))
    assert result == str(target)
    assert target.read_bytes() == b'cached'
    connect.assert_not_called()
