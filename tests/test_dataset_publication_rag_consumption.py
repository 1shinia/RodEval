"""Exercise native MTEB resolution for every publishable installed text type.

No model, remote dataset or production catalogue is loaded. Local cache loading
is simulated at the task boundary; task resolution uses real installed classes.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import mteb
import pytest

from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig
from evalscope.backend.rag_eval.mteb import task_template as runner
from evalscope.service import dataset_catalog_mteb as snapshots


def test_every_installed_text_task_type_uses_native_resolution(monkeypatch):
    representatives = {}
    for task in mteb.get_tasks(modalities=['text'], exclusive_modality_filter=True, exclude_aggregate=True):
        representatives.setdefault(task.metadata.type, task.metadata.name)
    assert {'Classification', 'Clustering', 'STS', 'Retrieval', 'Reranking'} <= representatives.keys()
    snapshot_load = Mock(side_effect=AssertionError('normal-user native resolution must not load snapshots'))
    monkeypatch.setattr(snapshots, 'load_published_tasks', snapshot_load)
    for task_type, name in representatives.items():
        tasks = list(runner.resolve_tasks(MTEBEvalConfig(
            task_names=[name], offline=True, allow_download=False)))
        assert [(task.metadata.name, task.metadata.type) for task in tasks] == [(name, task_type)]
        # All resolved classes enter the same native load_data preflight. It is
        # not restricted to the three types supported by internal snapshots.
        def loaded_task(task=tasks[0], kind=task_type):
            if kind == 'Reranking':
                task.dataset = {'default': {'test': {
                    'queries': {'id': ['q1']}, 'top_ranked': {'q1': ['d1']}}}}
        load = Mock(side_effect=loaded_task)
        monkeypatch.setattr(tasks[0], 'load_data', load)
        runner._preflight_offline_tasks(tasks, offline=True, allow_download=False)
        load.assert_called_once_with()
    snapshot_load.assert_not_called()


def test_native_cache_miss_reports_actionable_error_without_fallback():
    task = SimpleNamespace(metadata=SimpleNamespace(name='PublishedWithoutCache'),
                           load_data=Mock(side_effect=FileNotFoundError('cache absent')))
    with pytest.raises(RuntimeError, match='PublishedWithoutCache.*cache absent'):
        runner._preflight_offline_tasks([task], offline=True, allow_download=False)


def test_internal_snapshot_resolution_preserved(monkeypatch):
    tasks = [object()]
    load = Mock(return_value=tasks)
    monkeypatch.setattr(snapshots, 'load_published_tasks', load)
    selection = [{'id': 1, 'fingerprint': 'internal'}]
    assert runner.resolve_tasks(MTEBEvalConfig(
        task_names=['internal'], catalog_selection=selection)) is tasks
    load.assert_called_once_with(selection, ['internal'])
