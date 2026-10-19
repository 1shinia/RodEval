"""Temporary-only MTEB snapshot contract tests; no model or network is used."""
import json
import socket
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from evalscope.service import dataset_catalog as catalog
from evalscope.service import dataset_catalog_mteb as snapshots


def prepare_snapshot(task_name, destination):
    """Create legacy snapshot fixtures only; production no longer generates versions."""
    path = snapshots._path(destination)
    if path.exists():
        raise catalog.CatalogError('Snapshot destination already exists; never overwrite a version.')
    with snapshots.offline_guard():
        task = snapshots._get_task(task_name)
        identity = snapshots._identity(task)
        task.load_data()
        if not getattr(task, 'data_loaded', False):
            raise catalog.CatalogError('MTEB loader did not finish.')
        payload = {'format': snapshots.SOURCE, 'task': identity, 'versions': snapshots._versions(), 'dataset': snapshots._encode(task.dataset)}
        # Decode now to exercise the exact runner consumption path.
        snapshots._validate_dataset(identity['type'], snapshots._decode(payload['dataset']))
        data = snapshots._json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.mteb-', dir=path.parent))
    try:
        (temporary / 'snapshot.json').write_bytes(data)
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    fingerprint = snapshots._digest(data)
    return {'source': snapshots.SOURCE, 'cache_path': str(path), 'fingerprint': fingerprint,
            'details': {'validator': snapshots.SOURCE, 'snapshot_fingerprint': fingerprint,
                        'task_type': identity['type'], 'versions': payload['versions']}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, 'known_cache_roots', lambda: [tmp_path])
    calls = []

    class Task:
        metadata = SimpleNamespace(name='FixtureSTS', type='STS', modalities=['text'])

        def load_data(self, **kwargs):
            calls.append(True)
            self.dataset = {'test': Dataset.from_list([{'sentence1': 'a', 'sentence2': 'b', 'score': 1.0}])}
            self.data_loaded = True

    monkeypatch.setattr(snapshots, '_get_task', lambda name: Task())
    monkeypatch.setattr(snapshots, '_versions', lambda: {'mteb': 'fixture', 'datasets': 'fixture'})
    return tmp_path / 'snapshot', calls


def test_roundtrip_reloads_fixed_data_without_original_loader(env):
    path, calls = env
    receipt = prepare_snapshot('FixtureSTS', path)
    assert len(calls) == 1
    task = snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureSTS')
    task.dataset['test'] = task.dataset['test'].select([])
    task.data_loaded = False
    task.load_data()
    assert task.dataset['test'][0]['sentence1'] == 'a'
    assert task.data_loaded is True
    assert len(calls) == 1


def test_tampering_is_rejected(env):
    path, _ = env
    receipt = prepare_snapshot('FixtureSTS', path)
    data = path / 'snapshot.json'
    data.write_text(data.read_text().replace('sentence1', 'sentenceX'))
    with pytest.raises(catalog.CatalogError, match='fingerprint'):
        snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureSTS')


def test_missing_file_and_wrong_task_are_rejected(env):
    path, _ = env
    receipt = prepare_snapshot('FixtureSTS', path)
    with pytest.raises(catalog.CatalogError):
        snapshots.load_snapshot(path, receipt['fingerprint'], 'OtherTask')
    (path / 'snapshot.json').unlink()
    with pytest.raises((catalog.CatalogError, FileNotFoundError)):
        snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureSTS')


def test_no_overwrite_and_no_path_escape(env):
    path, _ = env
    prepare_snapshot('FixtureSTS', path)
    with pytest.raises(catalog.CatalogError):
        prepare_snapshot('FixtureSTS', path)
    with pytest.raises(catalog.CatalogError):
        prepare_snapshot('FixtureSTS', path.parent / '..' / 'escape')


def test_network_is_blocked_and_environment_restored(monkeypatch):
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    import os
    with snapshots.offline_guard():
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        with pytest.raises(catalog.CatalogError):
            socket.create_connection(('example.invalid', 443))
    assert 'HF_HUB_OFFLINE' not in os.environ


def test_symlink_and_extra_files_rejected(env):
    path, _ = env
    receipt = prepare_snapshot('FixtureSTS', path)
    (path / 'extra.pkl').write_bytes(b'untrusted')
    with pytest.raises(catalog.CatalogError):
        snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureSTS')
    (path / 'extra.pkl').unlink()
    data = path / 'snapshot.json'
    original = data.read_bytes()
    data.unlink()
    target = path.parent / 'target.json'
    target.write_bytes(original)
    data.symlink_to(target)
    with pytest.raises(catalog.CatalogError):
        snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureSTS')


def test_runtime_version_change_rejected(env, monkeypatch):
    path, _ = env
    receipt = prepare_snapshot('FixtureSTS', path)
    monkeypatch.setattr(snapshots, '_versions', lambda: {'mteb': 'other', 'datasets': 'fixture'})
    with pytest.raises(catalog.CatalogError, match='version'):
        snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureSTS')


def test_unsupported_task_and_media_fail_closed(env, monkeypatch):
    path, _ = env
    task = SimpleNamespace(metadata=SimpleNamespace(name='FixtureSTS', type='ImageText', modalities=['image']))
    monkeypatch.setattr(snapshots, '_get_task', lambda name: task)
    with pytest.raises(catalog.CatalogError):
        prepare_snapshot('FixtureSTS', path)
    assert not path.exists()


def test_published_runner_resolution_uses_server_entry(env, monkeypatch):
    path, _ = env
    receipt = prepare_snapshot('FixtureSTS', path)
    entry = {'benchmark': 'FixtureSTS', 'category': 'rag', 'source': snapshots.SOURCE,
             'cache_path': str(path), 'details': {'snapshot_fingerprint': receipt['fingerprint']}}
    selections = [{'id': 1, 'fingerprint': 'a' * 64}]
    monkeypatch.setattr(catalog, 'resolve_selection', lambda ident, fingerprint: entry)
    tasks = snapshots.load_published_tasks(selections, ['FixtureSTS'])
    assert tasks[0].dataset['test'].num_rows == 1
    with pytest.raises(catalog.CatalogError):
        snapshots.load_published_tasks(selections, ['OtherTask'])


def test_empty_dataset_rejected(env, monkeypatch):
    path, _ = env
    class EmptyTask:
        metadata = SimpleNamespace(name='FixtureSTS', type='STS', modalities=['text'])
        def load_data(self):
            self.data_loaded = True
            self.dataset = {'test': Dataset.from_list([])}
    monkeypatch.setattr(snapshots, '_get_task', lambda name: EmptyTask())
    with pytest.raises(catalog.CatalogError):
        prepare_snapshot('FixtureSTS', path)
    assert not path.exists()


@pytest.mark.parametrize('task_type', ['Retrieval', 'Reranking'])
def test_retrieval_complete_dependency_roundtrip(env, monkeypatch, task_type):
    path, _ = env
    class RetrievalTask:
        metadata = SimpleNamespace(name='FixtureRetrieval', type=task_type, modalities=['text'])
        def load_data(self):
            self.dataset = {'default': {'test': {
                'corpus': Dataset.from_list([{'id': 'd1', 'text': 'document'}]),
                'queries': Dataset.from_list([{'id': 'q1', 'text': 'query'}]),
                'relevant_docs': {'q1': {'d1': 1}},
                'top_ranked': {'q1': ['d1']},
            }}}
            self.data_loaded = True
    monkeypatch.setattr(snapshots, '_get_task', lambda name: RetrievalTask())
    receipt = prepare_snapshot('FixtureRetrieval', path)
    restored = snapshots.load_snapshot(path, receipt['fingerprint'], 'FixtureRetrieval')
    split = restored.dataset['default']['test']
    assert split['relevant_docs'] == {'q1': {'d1': 1}}
    assert split['corpus'][0]['text'] == 'document'
    assert split['top_ranked'] == {'q1': ['d1']}


def test_catalogue_publish_resolve_revoke_and_worker_consumption(env, monkeypatch):
    import sqlite3
    from evalscope.service import db
    path, _ = env
    prepare_snapshot('FixtureSTS', path)
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    for version, _, sql in db._MIGRATIONS:
        if version in (24, 25):
            conn.executescript(sql)
    monkeypatch.setattr(db, '_get_conn', lambda: conn)
    original_check = catalog._check_cache
    monkeypatch.setattr(catalog, '_check_cache', lambda row, path: snapshots.check_cache(row, path)
                        if row['source'] == snapshots.SOURCE else original_check(row, path))
    monkeypatch.setattr(catalog, '_dependencies', lambda row: [])
    original_resolved = catalog._resolved
    monkeypatch.setattr(catalog, '_resolved', lambda row: catalog._item(row, True)
                        if row['source'] == snapshots.SOURCE else original_resolved(row))
    conn.execute('''INSERT INTO dataset_catalog
        (benchmark,category,source,dataset_id,subset,split,cache_path,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?)''',
        ('FixtureSTS', 'rag', snapshots.SOURCE, 'FixtureSTS', 'default', 'all', str(path), 'now', 'now'))
    conn.commit()
    try:
        assert catalog.validate(1)['item']['status'] == 'available'
        catalog.publish(1, True)
        public = catalog.list_entries()['items'][0]
        selection = [{'id': public['id'], 'fingerprint': public['fingerprint']}]
        task = snapshots.load_published_tasks(selection, ['FixtureSTS'])[0]
        assert task.dataset['test'].num_rows == 1
        catalog.publish(1, False)
        with pytest.raises(catalog.CatalogError):
            snapshots.load_published_tasks(selection, ['FixtureSTS'])
        catalog.publish(1, True)
        (path / 'snapshot.json').write_text('{}')
        assert catalog.list_entries()['items'] == []
        with pytest.raises(catalog.CatalogError):
            snapshots.load_published_tasks(selection, ['FixtureSTS'])
    finally:
        conn.close()
