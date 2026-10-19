"""Offline LLM/VLM input media must not trigger network downloads."""
from unittest.mock import Mock

import pytest

from evalscope.utils.url_utils import download_url, file_as_data


def test_offline_url_download_reuses_local_file_without_head(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    target = tmp_path / 'data.bin'
    target.write_bytes(b'cached')
    head = Mock(side_effect=AssertionError('must not contact origin'))
    get = Mock(side_effect=AssertionError('must not download'))
    monkeypatch.setattr('requests.head', head)
    monkeypatch.setattr('requests.get', get)
    download_url('https://example.invalid/data', str(target), num_retries=1)
    assert target.read_bytes() == b'cached'
    head.assert_not_called()
    get.assert_not_called()


def test_scicode_cached_attachment_does_not_use_network(monkeypatch, tmp_path):
    from evalscope.benchmarks.scicode.scicode_adapter import SciCodeAdapter
    from evalscope.api.benchmark import MultiTurnAdapter

    (tmp_path / 'test_data.h5').write_bytes(b'cached')
    adapter = object.__new__(SciCodeAdapter)
    adapter.docker_path = tmp_path
    monkeypatch.setattr(MultiTurnAdapter, 'load', lambda self: 'cached-dataset')
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    get = Mock(side_effect=AssertionError('must not download'))
    head = Mock(side_effect=AssertionError('must not contact remote host'))
    monkeypatch.setattr('requests.get', get)
    monkeypatch.setattr('requests.head', head)
    assert adapter.load() == 'cached-dataset'
    get.assert_not_called()
    head.assert_not_called()


def test_offline_url_download_rejects_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    get = Mock(side_effect=AssertionError('must not download'))
    monkeypatch.setattr('requests.get', get)
    with pytest.raises(RuntimeError, match='管理员'):
        download_url('https://example.invalid/data', str(tmp_path / 'missing'), num_retries=1)
    get.assert_not_called()


def test_offline_input_media_rejects_remote_fetch(monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    client = Mock(side_effect=AssertionError('must not fetch remote dataset media'))
    monkeypatch.setattr('httpx.Client', client)
    with pytest.raises(RuntimeError, match='管理员'):
        file_as_data('https://example.invalid/image.png')
    client.assert_not_called()


def test_offline_input_media_still_reads_cached_file(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    target = tmp_path / 'image.png'
    target.write_bytes(b'cached')
    assert file_as_data(str(target)) == (b'cached', 'image/png')


def test_admin_input_media_keeps_remote_fetch(monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    client = Mock()
    client.get.return_value.content = b'image'
    monkeypatch.setattr('httpx.Client', lambda: client)
    assert file_as_data('https://example.invalid/image.png') == (b'image', 'image/png')
    client.get.assert_called_once()
