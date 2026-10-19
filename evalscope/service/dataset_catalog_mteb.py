"""Offline MTEB loaded-data snapshots for the existing dataset catalogue.

Only trusted installed MTEB task classes are instantiated. Snapshot bytes are JSON,
never Python objects/code. Paths are internal server configuration, not HTTP input.
Preparation must run in an isolated worker: offline_guard temporarily blocks socket
connections process-wide. Catalogue publication and authorization stay with callers.
"""
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from . import dataset_catalog as catalog

SOURCE = 'mteb-snapshot-v1'
SUPPORTED_TYPES = {'STS', 'Retrieval', 'Reranking'}
_FEATURE_TYPES = {'Value', 'ClassLabel', 'Sequence', 'List', 'LargeList'}


def _versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ('mteb', 'datasets')}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plain(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_plain(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item) for key, item in value.items()}
    raise catalog.CatalogError('Unsupported snapshot value; only finite JSON data is allowed.')


def _check_features(value: Any) -> None:
    if isinstance(value, dict):
        if '_type' in value and value['_type'] not in _FEATURE_TYPES:
            raise catalog.CatalogError('Unsupported feature: media and external dependencies are forbidden.')
        if value.get('_type') == 'Value' and not (
                value.get('dtype', '').startswith(('int', 'uint', 'float'))
                or value.get('dtype') in ('string', 'large_string', 'bool')):
            raise catalog.CatalogError('Only self-contained text/numeric features are supported.')
        for child in value.values():
            _check_features(child)
    elif isinstance(value, list):
        for child in value:
            _check_features(child)


def _encode(value: Any) -> dict:
    from datasets import Dataset, DatasetDict
    if isinstance(value, Dataset):
        if not len(value):
            raise catalog.CatalogError('Empty task dataset.')
        features = value.features.to_dict()
        _check_features(features)
        return {'kind': 'dataset', 'features': features, 'data': _plain(value.to_dict())}
    if isinstance(value, dict):
        if not value or not all(type(key) is str for key in value):
            raise catalog.CatalogError('Empty dataset mapping or non-string keys.')
        return {'kind': 'dataset_dict' if isinstance(value, DatasetDict) else 'mapping',
                'items': {key: _encode(child) for key, child in value.items()}}
    return {'kind': 'json', 'value': _plain(value)}


def _decode(value: dict) -> Any:
    from datasets import Dataset, DatasetDict, Features
    kind = value['kind']
    if kind == 'dataset':
        _check_features(value['features'])
        dataset = Dataset.from_dict(_plain(value['data']), features=Features.from_dict(value['features']))
        if not len(dataset):
            raise catalog.CatalogError('Empty task dataset.')
        return dataset
    if kind in ('mapping', 'dataset_dict'):
        items = {key: _decode(child) for key, child in value['items'].items()}
        return DatasetDict(items) if kind == 'dataset_dict' else items
    if kind == 'json':
        return _plain(value['value'])
    raise catalog.CatalogError('Unknown snapshot node.')


def _get_task(name: str) -> Any:
    import mteb
    tasks = list(mteb.get_tasks(tasks=[name]))
    if len(tasks) != 1 or tasks[0].metadata.name != name:
        raise catalog.CatalogError('Task name must identify one installed MTEB task.')
    return tasks[0]


def _identity(task: Any) -> dict:
    meta = task.metadata
    modalities = getattr(meta, 'modalities', ['text'])
    if meta.type not in SUPPORTED_TYPES or any(str(item).lower() != 'text' for item in modalities):
        raise catalog.CatalogError('Only text STS/Retrieval/Reranking tasks are supported.')
    return {'name': meta.name, 'type': meta.type}


@contextmanager
def offline_guard() -> Iterator[None]:
    """Block network connections and temporarily set live HF offline flags."""
    names = ('HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')
    previous = {key: os.environ.get(key) for key in names}
    flags = []
    for module_name in ('datasets.config', 'huggingface_hub.constants'):
        module = sys.modules.get(module_name)
        for key in ('HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE'):
            if module is not None and hasattr(module, key):
                flags.append((module, key, getattr(module, key)))
                setattr(module, key, True)

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise catalog.CatalogError('Network access forbidden while preparing/loading MTEB snapshot.')

    try:
        os.environ.update({key: '1' for key in names})
        with patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked), \
                patch.object(socket, 'create_connection', blocked):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for module, key, value in flags:
            setattr(module, key, value)


def _path(path: str | Path) -> Path:
    candidate = Path(path).absolute()
    if '..' in candidate.parts:
        raise catalog.CatalogError('Snapshot path escapes cache boundary.')
    candidate = catalog._safe_path(candidate)
    for parent in (candidate, *candidate.parents):
        if parent.is_symlink():
            raise catalog.CatalogError('Snapshot paths must not contain symlinks.')
    return candidate


def _validate_dataset(task_type: str, dataset: Any) -> None:
    from datasets import Dataset
    found = []

    def visit(node: Any) -> None:
        if isinstance(node, Dataset):
            found.append(node)
            if task_type == 'STS' and not {'sentence1', 'sentence2', 'score'} <= set(node.column_names):
                raise catalog.CatalogError('Incomplete STS dataset.')
        elif isinstance(node, dict):
            if task_type in ('Retrieval', 'Reranking') and 'corpus' in node:
                corpus, queries, qrels = node.get('corpus'), node.get('queries'), node.get('relevant_docs')
                if not isinstance(corpus, Dataset) or not isinstance(queries, Dataset) or not isinstance(qrels, dict):
                    raise catalog.CatalogError('Incomplete corpus/queries/relevant_docs.')
                if not {'id', 'text'} <= set(corpus.column_names) or not {'id', 'text'} <= set(queries.column_names):
                    raise catalog.CatalogError('Missing retrieval id/text columns.')
                docs, qids = set(corpus['id']), set(queries['id'])
                if not qrels or not set(qrels) <= qids or any(
                        not isinstance(scores, dict) or not scores or not set(scores) <= docs
                        for scores in qrels.values()):
                    raise catalog.CatalogError('Dangling or empty qrels.')
            for child in node.values():
                visit(child)
    visit(dataset)
    if not found:
        raise catalog.CatalogError('Snapshot contains no Dataset.')
    if task_type in ('Retrieval', 'Reranking'):
        def has_corpus(node: Any) -> bool:
            return isinstance(node, dict) and ('corpus' in node or any(has_corpus(v) for v in node.values()))
        if not has_corpus(dataset):
            raise catalog.CatalogError('Only MTEB 2.x corpus/queries retrieval layouts are supported.')


def _read_snapshot(path: str | Path, fingerprint: str, task_name: str) -> dict:
    root = _path(path)
    if {child.name for child in root.iterdir()} != {'snapshot.json'}:
        raise catalog.CatalogError('Unexpected snapshot files.')
    file = _path(root / 'snapshot.json')
    data = file.read_bytes()
    if not isinstance(fingerprint, str) or _digest(data) != fingerprint:
        raise catalog.CatalogError('Snapshot fingerprint mismatch.')
    payload = json.loads(data)
    if payload['format'] != SOURCE or payload['task']['name'] != task_name:
        raise catalog.CatalogError('Snapshot task/format mismatch.')
    if payload['versions'] != _versions():
        raise catalog.CatalogError('Snapshot runtime version mismatch; prepare a new version.')
    return payload


def load_snapshot(path: str | Path, fingerprint: str, task_name: str) -> Any:
    """Restore a trusted installed task; all subsequent reloads use verified JSON.

    fingerprint must come from the server's published catalogue, never a path or
    digest supplied directly by an evaluation client. No native cache fallback.
    """
    with offline_guard():
        payload = _read_snapshot(path, fingerprint, task_name)
        task = _get_task(task_name)
        if _identity(task) != payload['task']:
            raise catalog.CatalogError('Installed task identity changed.')

    def fixed_load(self: Any, **kwargs: Any) -> None:
        if getattr(self, 'data_loaded', False):
            return
        current = _read_snapshot(path, fingerprint, task_name)
        dataset = _decode(current['dataset'])
        _validate_dataset(current['task']['type'], dataset)
        self.dataset = dataset
        self.data_loaded = True

    task.load_data = types.MethodType(fixed_load, task)
    task.data_loaded = False
    task.load_data()
    return task


def check_cache(row: dict, path: str | Path) -> tuple[str, dict]:
    """Catalogue _check_cache hook, before generic benchmark/adapter dispatch."""
    if (row['category'] != 'rag' or row['source'] != SOURCE or row['subset'] != 'default'
            or row['split'] != 'all' or row['revision'] is not None):
        raise catalog.CatalogError('Invalid MTEB catalogue identity.')
    root = _path(path)
    fingerprint = _digest((root / 'snapshot.json').read_bytes())
    task = load_snapshot(root, fingerprint, row['benchmark'])
    return 'available', {'validator': 'audited-cache-v2', 'snapshot_fingerprint': fingerprint,
                         'task_type': task.metadata.type}


def load_published_tasks(selections: list[dict], task_names: list[str]) -> list[Any]:
    """Minimal runner hook: resolve IDs+manifest fingerprints again at execution."""
    if not selections or not task_names or len(task_names) != len(set(task_names)):
        raise catalog.CatalogError('Explicit unique MTEB task names and published selections required.')
    entries = []
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {'id', 'fingerprint'}:
            raise catalog.CatalogError('Invalid catalogue selection.')
        entry = catalog.resolve_selection(selection['id'], selection['fingerprint'])
        if entry['category'] != 'rag' or entry['source'] != SOURCE:
            raise catalog.CatalogError('Selection is not a published MTEB snapshot.')
        entries.append(entry)
    if len(entries) != len(task_names) or {entry['benchmark'] for entry in entries} != set(task_names):
        raise catalog.CatalogError('Published versions do not match requested task names.')
    by_name = {entry['benchmark']: entry for entry in entries}
    return [load_snapshot(by_name[name]['cache_path'], by_name[name]['details']['snapshot_fingerprint'], name)
            for name in task_names]
