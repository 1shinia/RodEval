"""Offline tests for retrieval/reranking query sampling and aligned metadata."""
from types import SimpleNamespace

import pytest
from datasets import Dataset

from evalscope.backend.rag_eval.mteb.task_template import _apply_retrieval_limits


def make_task(kind):
    queries = Dataset.from_dict({'id': [str(i) for i in range(30)], 'text': [f'q{i}' for i in range(30)]})
    corpus = Dataset.from_dict({'id': ['doc'], 'text': ['content']})
    split = {
        'queries': queries,
        'corpus': corpus,
        'relevant_docs': {str(i): {'doc': 1} for i in range(30)},
    }
    if kind == 'Reranking':
        split['top_ranked'] = {str(i): [f'candidate-{i}'] for i in range(30)}
    return SimpleNamespace(metadata=SimpleNamespace(name=kind, type=kind), dataset={'default': {'test': split}})


@pytest.mark.parametrize('kind', ['Retrieval', 'Reranking'])
def test_random_queries_match_seeded_shuffle_and_keep_metadata_aligned(kind):
    first, second = make_task(kind), make_task(kind)
    original = first.dataset['default']['test']
    corpus = original['corpus']
    expected = list(original['queries'].shuffle(seed=42).select(range(5))['id'])
    assert expected != ['0', '1', '2', '3', '4']
    for task in (first, second):
        _apply_retrieval_limits(task, 5, True)
        split = task.dataset['default']['test']
        assert list(split['queries']['id']) == expected
        assert set(split['relevant_docs']) == set(expected)
        if kind == 'Reranking':
            assert split['top_ranked'] == {qid: [f'candidate-{qid}'] for qid in expected}
    assert first.dataset['default']['test']['corpus'] is corpus


@pytest.mark.parametrize('kind', ['Retrieval', 'Reranking'])
def test_sequential_queries_and_metadata_still_use_first_n(kind):
    task = make_task(kind)
    _apply_retrieval_limits(task, 2, False)
    split = task.dataset['default']['test']
    assert list(split['queries']['id']) == ['0', '1']
    assert set(split['relevant_docs']) == {'0', '1'}
    if kind == 'Reranking':
        assert set(split['top_ranked']) == {'0', '1'}


@pytest.mark.parametrize('limit', [30, 40])
@pytest.mark.parametrize('random_sample', [False, True])
def test_limit_covering_all_queries_preserves_data(limit, random_sample):
    task = make_task('Reranking')
    split = task.dataset['default']['test']
    queries = split['queries']
    _apply_retrieval_limits(task, limit, random_sample)
    assert split['queries'] is queries
    assert len(split['relevant_docs']) == 30
    assert len(split['top_ranked']) == 30
