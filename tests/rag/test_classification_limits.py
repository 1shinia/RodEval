"""Offline regression coverage for MTEB classification sample limits."""
from types import SimpleNamespace

from datasets import Dataset, DatasetDict

from evalscope.backend.rag_eval.mteb.task_template import _apply_classification_limits
import pytest


@pytest.mark.parametrize('random_sample', [False, True])
def test_nested_classification_limits_are_reproducible(random_sample):
    def make_task():
        return SimpleNamespace(
            metadata=SimpleNamespace(name='nested'), train_split='training',
            dataset={'eng': DatasetDict(
                training=Dataset.from_dict({'text': ['a', 'b', 'c'], 'label': [0, 0, 1]}),
                test=Dataset.from_dict({'text': list('defghijk'), 'label': [0, 1] * 4}),
            )},
        )
    first, second = make_task(), make_task()
    _apply_classification_limits(first, 2, random_sample)
    _apply_classification_limits(second, 2, random_sample)
    assert len(first.dataset['eng']['training']) == 3
    assert len(first.dataset['eng']['test']) == 2
    assert first.dataset['eng']['test']['text'] == second.dataset['eng']['test']['text']


def test_cross_validation_keeps_training_folds_intact():
    dataset = DatasetDict(test=Dataset.from_dict({'text': list('abcd'), 'label': [0, 0, 1, 1]}))
    task = SimpleNamespace(metadata=SimpleNamespace(name='cv'), dataset=dataset, is_cross_validation=True)
    _apply_classification_limits(task, 1)
    assert task.dataset is dataset


def test_one_stage_dispatch_preserves_training_and_other_task_limits(tmp_path, monkeypatch):
    from evalscope.backend.rag_eval.mteb import task_template as module
    from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig
    from unittest.mock import Mock
    def task(kind):
        return SimpleNamespace(
            metadata=SimpleNamespace(name=kind, type=kind), train_split='train',
            load_data=lambda: None,
            dataset=DatasetDict(
                train=Dataset.from_dict({'text': list('abcd'), 'label': [0, 0, 1, 1]}),
                test=Dataset.from_dict({'text': list('efgh'), 'label': [0, 1, 0, 1]}),
            ),
        )
    classification, other = task('Classification'), task('STS')
    monkeypatch.setattr(module, 'load_model', lambda _: SimpleNamespace())
    monkeypatch.setattr(module, 'resolve_tasks', lambda _: [classification, other])
    monkeypatch.setattr(module, '_preflight_offline_tasks', lambda *args: None)
    monkeypatch.setattr(module, 'show_results', lambda *args: None)
    monkeypatch.setattr(module, '_collect_perf_metrics', lambda *args: None)
    evaluate = Mock(return_value=SimpleNamespace(task_results=[]))
    monkeypatch.setattr(module.mteb, 'evaluate', evaluate)
    module.one_stage_eval(None, MTEBEvalConfig(task_names=['example'], limits=2, output_folder=str(tmp_path)))
    assert len(classification.dataset['train']) == 4
    assert len(classification.dataset['test']) == 2
    assert len(other.dataset['train']) == 2
    assert len(other.dataset['test']) == 2
    evaluate.assert_called_once()



def test_classification_limits_preserve_training_classes():
    train = Dataset.from_dict({'text': ['a', 'b', 'c', 'd'], 'label': [0, 0, 1, 1]})
    test = Dataset.from_dict({'text': ['e', 'f', 'g'], 'label': [0, 1, 0]})
    task = SimpleNamespace(
        metadata=SimpleNamespace(name='example'), train_split='train',
        dataset=DatasetDict(train=train, test=test),
    )
    _apply_classification_limits(task, 2, False)
    assert task.dataset['train'] is train
    assert task.dataset['train']['label'] == [0, 0, 1, 1]
    assert task.dataset['test']['text'] == ['e', 'f']
    # The actual sklearn classifier must still be trainable after limiting.
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression().fit([[0], [0.1], [1], [1.1]], task.dataset['train']['label'])
    assert set(classifier.classes_) == {0, 1}
