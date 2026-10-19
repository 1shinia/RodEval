"""Cached attachment reuse must not call online snapshot preparation."""
from unittest.mock import Mock

import pytest

from evalscope.api.dataset.hub import download_dataset_file


@pytest.mark.parametrize('source', ['modelscope', 'huggingface'])
def test_offline_attachment_uses_local_only_sdk(source, tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    cached = tmp_path / 'media.zip'
    cached.write_bytes(b'cached')
    sdk = Mock(return_value=str(cached))
    if source == 'modelscope':
        import importlib
        module = importlib.import_module('modelscope.hub.file_download')
        monkeypatch.setattr(module, 'dataset_file_download', sdk)
    else:
        monkeypatch.setattr('huggingface_hub.hf_hub_download', sdk)
    assert download_dataset_file('audit/data', 'media.zip', source, force_redownload=True) == str(cached)
    assert sdk.call_args.kwargs['local_files_only'] is True
    assert not sdk.call_args.kwargs.get('force_download', False)


@pytest.mark.parametrize('source', ['modelscope', 'huggingface'])
def test_missing_offline_attachment_reports_admin_without_online_fallback(source, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    sdk = Mock(side_effect=FileNotFoundError('cache miss'))
    if source == 'modelscope':
        import importlib
        module = importlib.import_module('modelscope.hub.file_download')
        monkeypatch.setattr(module, 'dataset_file_download', sdk)
    else:
        monkeypatch.setattr('huggingface_hub.hf_hub_download', sdk)
    with pytest.raises(RuntimeError, match='管理员'):
        download_dataset_file('audit/data', 'media.zip', source)
    sdk.assert_called_once()
    assert sdk.call_args.kwargs['local_files_only'] is True


@pytest.mark.parametrize('source', ['modelscope', 'huggingface'])
def test_admin_preserves_online_attachment_behavior(source, tmp_path, monkeypatch):
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    target = tmp_path / 'media.zip'
    target.write_bytes(b'prepared')
    sdk = Mock(return_value=str(tmp_path) if source == 'modelscope' else str(target))
    monkeypatch.setattr('modelscope.dataset_snapshot_download' if source == 'modelscope'
                        else 'huggingface_hub.hf_hub_download', sdk)
    assert download_dataset_file('audit/data', 'media.zip', source, force_redownload=True) == str(target)
    sdk.assert_called_once()
    assert not sdk.call_args.kwargs.get('local_files_only', False)
    if source == 'huggingface':
        assert sdk.call_args.kwargs['force_download'] is True
