"""Offline regression coverage for native benchmark sample selection."""
import json
import random
from typing import Any
from types import SimpleNamespace

import pytest
from datasets import Dataset as HFDataset

from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.dataset.loader import LocalDataLoader, RemoteDataLoader
from evalscope.benchmarks.gsm8k.gsm8k_adapter import GSM8KAdapter


def make_adapter(path, *, enabled=False, limit=3, seed=42, shuffle=False, repeats=1) -> Any:
    adapter = object.__new__(GSM8KAdapter)
    adapter._task_config = SimpleNamespace(
        limit=limit, random_sample=enabled, seed=seed, repeats=repeats,
        dataset_hub='local', dataset_dir=str(path.parent / 'cache'),
    )
    adapter._benchmark_meta = SimpleNamespace(
        dataset_id=str(path), eval_split='test', train_split='train', default_subset='default',
        subset_list=['main'], shuffle=shuffle, shuffle_choices=False, force_redownload=False,
        few_shot_num=3, few_shot_random=False,
    )
    adapter.reformat_subset = False
    adapter.split_as_subset = False
    return adapter


@pytest.fixture
def records(tmp_path):
    rows = [{'question': f'q{i}', 'answer': f'work #### {i}'} for i in range(30)]
    path = tmp_path / 'main_test.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    return path, rows


@pytest.mark.parametrize('loader_type', [LocalDataLoader, RemoteDataLoader])
def test_random_selection_before_limit_is_seeded(records, monkeypatch, loader_type):
    path, rows = records
    if loader_type is RemoteDataLoader:
        import evalscope.api.dataset.loader as module
        monkeypatch.setattr(module.DatasetHub, 'load', lambda *a, **kw: HFDataset.from_list(rows))
    def selected(seed):
        adapter = make_adapter(path, enabled=True, seed=seed)
        return [sample.input for sample in adapter.load_subset('main', loader_type)]
    first = selected(42)
    assert first != ['q0', 'q1', 'q2']
    assert selected(42) == first
    assert selected(7) != first


def test_sequential_repeats_and_fewshot_are_preserved(records):
    path, _ = records
    adapter = make_adapter(path, enabled=False, repeats=2)
    assert [s.input for s in adapter.load_subset('main', LocalDataLoader)] == [
        'q0', 'q0', 'q1', 'q1', 'q2', 'q2']
    adapter._task_config.random_sample = True
    assert [s.input for s in adapter.load_fewshot_subset('main', LocalDataLoader)] == ['q0', 'q1', 'q2']
    adapter._task_config.limit = None
    adapter._task_config.repeats = 1
    assert [s.input for s in adapter.load_subset('main', LocalDataLoader)] == [f'q{i}' for i in range(30)]


def test_existing_dataset_shuffle_is_preserved(records):
    path, _ = records
    adapter = make_adapter(path, shuffle=True)
    assert [s.input for s in adapter.load_subset('main', LocalDataLoader)] != ['q0', 'q1', 'q2']


@pytest.mark.parametrize('seed', [42, 7])
def test_reformatted_subsets_use_task_seed_and_keep_fewshot_separate(records, seed):
    path, _ = records
    adapter = make_adapter(path, enabled=True, seed=seed)
    adapter.reformat_subset = True
    adapter._benchmark_meta.subset_list = ['a', 'b']
    samples = [Sample(input=f'{key}{i}', subset_key=key) for key in ['a', 'b'] for i in range(30)]

    def load_func(_):
        return MemoryDataset(samples, name='offline')

    selected = adapter.load_subsets(load_func)
    for key in ['a', 'b']:
        expected = [f'{key}{i}' for i in range(30)]
        random.Random(seed).shuffle(expected)
        assert [s.input for s in selected[key]] == expected[:3]
    fewshot = adapter.load_subsets(load_func, is_fewshot=True)
    for key in ['a', 'b']:
        assert [s.input for s in fewshot[key]] == [f'{key}{i}' for i in range(3)]
