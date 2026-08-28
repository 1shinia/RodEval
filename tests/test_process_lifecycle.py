"""Task registry slot must be released on every run_in_subprocess exit path.

Regression guard for the direct-mode /eval/launch slot leak: previously a task
that failed during spawn/IPC/result-decode kept its registry slot forever,
permanently consuming a concurrent-task slot.
"""
import multiprocessing

import pytest

from evalscope.service.utils import process


def _boom():
    raise RuntimeError('boom')


def _ok():
    return 42


def _running_ids():
    return [t['task_id'] for t in process.get_running_tasks()]


def test_slot_released_when_child_raises():
    task_id = 'eval_slot_raise'
    process.register_process(task_id, multiprocessing.Process(target=lambda: None), task_type='eval', model='m')
    assert task_id in _running_ids()

    with pytest.raises(RuntimeError, match='boom'):
        process.run_in_subprocess(_boom, task_id=task_id, task_type='eval', model='m')

    assert task_id not in _running_ids()
    # Idempotent: unregistering again must not raise (background cleanup paths
    # call it unconditionally in finally blocks).
    process.unregister_process(task_id)


def test_slot_released_on_success():
    task_id = 'eval_slot_ok'
    process.register_process(task_id, multiprocessing.Process(target=lambda: None), task_type='eval', model='m')

    assert process.run_in_subprocess(_ok, task_id=task_id, task_type='eval', model='m') == 42

    assert task_id not in _running_ids()


def test_slot_released_when_register_placeholder_absent():
    # run_in_subprocess with a task_id that was never registered must still
    # clean up cleanly (finalize_slot skips missing placeholders).
    task_id = 'eval_slot_never_registered'
    process.run_in_subprocess(_ok, task_id=task_id, task_type='eval', model='m')
    assert task_id not in _running_ids()
