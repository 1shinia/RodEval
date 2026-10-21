"""Offline MTEB config discovery uses only the selected repository/revision cache."""
import socket
from unittest.mock import Mock

from types import SimpleNamespace
import pytest
from datasets import config as datasets_config
from mteb.abstasks import retrieval_dataset_loaders as loaders

from evalscope.backend.rag_eval.mteb.task_template import _offline_dataset_guard


def cache_config(root, config, revision='revision-a', repo='mteb___fixture_rr'):
    path = root / repo / config / '0.0.0' / revision
    path.mkdir(parents=True)
    (path / 'dataset_info.json').write_text('{}')
    (path / 'fixture_rr-test.arrow').write_bytes(b'fixture')
    return path


def test_offline_discovers_cached_candidates_for_exact_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_config, 'HF_DATASETS_CACHE', tmp_path)
    original = Mock(return_value=['default'])
    monkeypatch.setattr(loaders, 'get_dataset_config_names', original)
    connect = Mock(side_effect=AssertionError('network forbidden'))
    monkeypatch.setattr(socket.socket, 'connect', connect)
    for config in ['default', 'corpus', 'queries', 'top_ranked', 'en-top_ranked']:
        cache_config(tmp_path, config)
    cache_config(tmp_path, 'other-revision-top_ranked', revision='revision-b')
    cache_config(tmp_path, 'other-repo-top_ranked', repo='mteb___other_rr')
    empty = tmp_path / 'mteb___fixture_rr' / 'incomplete' / '0.0.0' / 'revision-a'
    empty.mkdir(parents=True)
    (empty / 'dataset_info.json').write_text('{}')
    with _offline_dataset_guard(True):
        configs = loaders.get_dataset_config_names('mteb/FixtureRR', 'revision-a')
    assert set(configs) == {'default', 'corpus', 'queries', 'top_ranked', 'en-top_ranked'}
    assert loaders.get_dataset_config_names is original
    connect.assert_not_called()


@pytest.mark.parametrize('candidates', [None, {}, {'other-query': ['d1']}, {'q1': []}])
def test_offline_preflight_rejects_incomplete_reranking_candidates(candidates):
    from evalscope.backend.rag_eval.mteb.task_template import _preflight_offline_tasks
    task = SimpleNamespace(
        metadata=SimpleNamespace(name='MissingCandidates', type='Reranking'),
        dataset={'default': {'test': {'queries': {'id': ['q1']}, 'top_ranked': candidates}}},
        load_data=Mock())
    with pytest.raises(RuntimeError, match='MissingCandidates.*top_ranked'):
        _preflight_offline_tasks([task], offline=True, allow_download=False)


def test_online_discovery_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_config, 'HF_DATASETS_CACHE', tmp_path)
    cache_config(tmp_path, 'top_ranked')
    original = Mock(return_value=['default'])
    monkeypatch.setattr(loaders, 'get_dataset_config_names', original)
    with _offline_dataset_guard(False):
        assert loaders.get_dataset_config_names('mteb/FixtureRR', 'revision-a') == ['default']
    assert loaders.get_dataset_config_names is original
