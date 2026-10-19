"""Special dataset preparation must preserve offline cache-only access."""
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

CASES = [('frames', 'FramesAdapter', 'test.jsonl'),
         ('maritime_ocr_bench', 'MaritimeOCRBenchAdapter', 'maritime_ocr_bench.jsonl')]


def harness(name, cls, monkeypatch):
    module = importlib.import_module(f'evalscope.benchmarks.{name}.{name}_adapter')
    if hasattr(module, 'check_import'):
        monkeypatch.setattr(module, 'check_import', Mock())
    obj = SimpleNamespace(dataset_id='audit/nonexistent', eval_split='test',
                          record_to_sample=Mock(), limit=None, repeats=1, shuffle=False, subset_list=['IE'])
    obj._resolve_dataset_file = lambda path: getattr(module, cls)._resolve_dataset_file(obj, path)
    monkeypatch.setattr(module, 'LocalDataLoader', Mock(return_value=SimpleNamespace(load=lambda: [])))
    if name == 'maritime_ocr_bench':
        monkeypatch.setattr(module, 'DatasetDict', SimpleNamespace(from_dataset=lambda **kw: {}))
    return module, obj


@pytest.mark.parametrize('name,cls,filename', CASES)
def test_missing_snapshot_never_downloads(name, cls, filename, monkeypatch):
    module, obj = harness(name, cls, monkeypatch)
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    online = Mock(side_effect=AssertionError('online download'))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', online)
    monkeypatch.setattr(importlib.import_module('modelscope.hub.snapshot_download'),
                        'snapshot_download', Mock(side_effect=FileNotFoundError()))
    with pytest.raises(RuntimeError, match='管理员'):
        getattr(module, cls).load(obj)
    online.assert_not_called()


@pytest.mark.parametrize('name,cls,filename', CASES)
@pytest.mark.parametrize('offline', [True, False])
def test_snapshot_cache_and_admin_behavior(name, cls, filename, offline, tmp_path, monkeypatch):
    module, obj = harness(name, cls, monkeypatch)
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1' if offline else '0')
    (tmp_path / filename).write_text('{}\n')
    online, local = Mock(return_value=str(tmp_path)), Mock(return_value=str(tmp_path))
    monkeypatch.setattr('modelscope.dataset_snapshot_download', online)
    monkeypatch.setattr(importlib.import_module('modelscope.hub.snapshot_download'), 'snapshot_download', local)
    getattr(module, cls).load(obj)
    if offline:
        online.assert_not_called()
        assert local.call_args.kwargs['local_files_only'] is True
        assert local.call_args.kwargs['repo_type'] == 'dataset'
    else:
        online.assert_called_once()
        local.assert_not_called()
    if name == 'frames':
        assert (local if offline else online).call_args.kwargs['allow_file_pattern'] == 'test.jsonl'


@pytest.mark.parametrize('name,cls,filename', CASES)
def test_incomplete_local_cache_reports_admin(name, cls, filename, tmp_path, monkeypatch):
    module, obj = harness(name, cls, monkeypatch)
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    obj.dataset_id = str(tmp_path)
    with pytest.raises(RuntimeError, match='管理员'):
        getattr(module, cls).load(obj)


@pytest.mark.parametrize('empty', [False, True])
def test_aa_missing_documents_never_downloads(tmp_path, monkeypatch, empty):
    from evalscope.benchmarks.aa_lcr import aa_lcr_adapter as module
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    monkeypatch.setattr(module, 'DEFAULT_EVALSCOPE_CACHE_DIR', str(tmp_path))
    if empty:
        (tmp_path / module.DEFAULT_CACHE_SUBDIR / module.DEFAULT_EXTRACTED_DIR_NAME).mkdir(parents=True)
    online = Mock(side_effect=AssertionError('online download'))
    monkeypatch.setattr(module.urllib.request, 'urlretrieve', online)
    with pytest.raises(RuntimeError, match='管理员'):
        module.AALCRAdapter._ensure_text_dir_downloaded(SimpleNamespace())
    online.assert_not_called()


@pytest.mark.parametrize('state', ['missing', 'empty', 'valid'])
def test_aa_record_documents(tmp_path, monkeypatch, state):
    from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1')
    folder = tmp_path / 'a' / 'b'
    if state != 'missing':
        folder.mkdir(parents=True)
        (folder / 'doc.txt').write_text('cached document' if state == 'valid' else '')
    obj = SimpleNamespace(text_dir=tmp_path)
    record = {'document_category': 'a', 'document_set_id': 'b'}
    if state == 'valid':
        assert 'cached document' in AALCRAdapter._get_context(obj, record)
    else:
        with pytest.raises(RuntimeError, match='管理员'):
            AALCRAdapter._get_context(obj, record)


@pytest.mark.parametrize('offline', [True, False])
def test_aa_cached_documents_reused(tmp_path, monkeypatch, offline):
    from evalscope.benchmarks.aa_lcr import aa_lcr_adapter as module
    monkeypatch.setenv('EVALSCOPE_DATASET_OFFLINE', '1' if offline else '0')
    monkeypatch.setattr(module, 'DEFAULT_EVALSCOPE_CACHE_DIR', str(tmp_path))
    root = tmp_path / module.DEFAULT_CACHE_SUBDIR / module.DEFAULT_EXTRACTED_DIR_NAME
    root.mkdir(parents=True)
    (root / 'doc.txt').write_text('cached document')
    online = Mock(side_effect=AssertionError('online download'))
    monkeypatch.setattr(module.urllib.request, 'urlretrieve', online)
    assert module.AALCRAdapter._ensure_text_dir_downloaded(SimpleNamespace()) == root
    online.assert_not_called()


def test_aa_admin_still_downloads(tmp_path, monkeypatch):
    import zipfile
    from evalscope.benchmarks.aa_lcr import aa_lcr_adapter as module
    monkeypatch.delenv('EVALSCOPE_DATASET_OFFLINE', raising=False)
    monkeypatch.setattr(module, 'DEFAULT_EVALSCOPE_CACHE_DIR', str(tmp_path))
    def retrieve(url, path):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(f'{module.DEFAULT_EXTRACTED_DIR_NAME}/doc.txt', 'cached document')
    online = Mock(side_effect=retrieve)
    monkeypatch.setattr(module.urllib.request, 'urlretrieve', online)
    root = module.AALCRAdapter._ensure_text_dir_downloaded(SimpleNamespace())
    assert (root / 'doc.txt').read_text() == 'cached document'
    online.assert_called_once()
