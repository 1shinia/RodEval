"""Interrupted attachment downloads must not publish partial cache files."""
from unittest.mock import Mock

import pytest

from evalscope.utils.url_utils import download_url


def response(chunks, length):
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.headers = {'content-length': str(length)}
    result.iter_content.return_value = chunks
    return result


def test_short_download_never_publishes_target(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'
    monkeypatch.setattr('requests.get', Mock(return_value=response([b'part'], 100)))
    with pytest.raises(RuntimeError):
        download_url('https://example.invalid/data', str(target), num_retries=1)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_failed_refresh_preserves_existing_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'
    target.write_bytes(b'previous-complete')
    monkeypatch.setattr('requests.head', Mock(return_value=Mock(headers={'content-length': '100'})))
    monkeypatch.setattr('requests.get', Mock(return_value=response([b'part'], 100)))
    with pytest.raises(RuntimeError):
        download_url('https://example.invalid/data', str(target), num_retries=1)
    assert target.read_bytes() == b'previous-complete'
    assert list(tmp_path.iterdir()) == [target]


def test_offline_empty_cache_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    target = tmp_path / 'attachment.bin'
    target.touch()
    with pytest.raises(RuntimeError, match='管理员'):
        download_url('https://example.invalid/data', str(target))


def test_interrupted_stream_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'

    def chunks():
        yield b'part'
        raise ConnectionError('interrupted')

    monkeypatch.setattr('requests.get', Mock(return_value=response(chunks(), 100)))
    with pytest.raises(RuntimeError):
        download_url('https://example.invalid/data', str(target), num_retries=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('encoding', ['identity', 'Identity', ' identity '])
def test_identity_short_download_preserves_old_cache(tmp_path, monkeypatch, encoding):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'
    target.write_bytes(b'previous-complete')
    monkeypatch.setattr('requests.head', Mock(return_value=Mock(headers={'content-length': '100'})))
    reply = response([b'part'], 100)
    reply.headers['content-encoding'] = encoding
    monkeypatch.setattr('requests.get', Mock(return_value=reply))
    with pytest.raises(RuntimeError):
        download_url('https://example.invalid/data', str(target), num_retries=1)
    assert target.read_bytes() == b'previous-complete'
    assert list(tmp_path.iterdir()) == [target]


def test_encoded_response_does_not_compare_decoded_size(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'
    reply = response([b'decoded-content'], 8)
    reply.headers['content-encoding'] = 'gzip'
    monkeypatch.setattr('requests.get', Mock(return_value=reply))
    download_url('https://example.invalid/data', str(target), num_retries=1)
    assert target.read_bytes() == b'decoded-content'


def test_admin_matching_cache_keeps_head_fast_path(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'
    target.write_bytes(b'full')
    head = Mock(return_value=Mock(headers={'content-length': '4'}))
    get = Mock(side_effect=AssertionError('cache hit must not download'))
    monkeypatch.setattr('requests.head', head)
    monkeypatch.setattr('requests.get', get)
    download_url('https://example.invalid/data', str(target), num_retries=1)
    head.assert_called_once()
    get.assert_not_called()
    assert target.read_bytes() == b'full'


def test_retry_publishes_only_complete_attempt(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    monkeypatch.setattr('time.sleep', Mock())
    target = tmp_path / 'attachment.bin'
    get = Mock(side_effect=[response([b'bad'], 100), response([b'full'], 4)])
    monkeypatch.setattr('requests.get', get)
    download_url('https://example.invalid/data', str(target), num_retries=2)
    assert get.call_count == 2
    assert target.read_bytes() == b'full'
    assert list(tmp_path.iterdir()) == [target]


def test_complete_download_publishes_target(tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'attachment.bin'
    monkeypatch.setattr('requests.get', Mock(return_value=response([b'full'], 4)))
    download_url('https://example.invalid/data', str(target), num_retries=1)
    assert target.read_bytes() == b'full'
    assert list(tmp_path.iterdir()) == [target]
