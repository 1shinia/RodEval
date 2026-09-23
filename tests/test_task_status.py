from evalscope.service.task_status import (
    FAILED,
    PARTIAL_SUCCESS,
    STOPPED,
    COMPLETED,
    aggregate_batch_status,
    normalize_persisted_task_status,
    normalize_status,
)


def test_batch_status_aggregation_distinguishes_success_partial_and_failure():
    assert aggregate_batch_status(total=2, completed=2, errors=0) == COMPLETED
    assert aggregate_batch_status(total=2, completed=1, errors=1) == PARTIAL_SUCCESS
    assert aggregate_batch_status(total=2, completed=0, errors=2) == FAILED
    assert aggregate_batch_status(total=2, completed=1, errors=0) == 'running'
    assert aggregate_batch_status(total=2, completed=1, errors=1, stopped=True) == STOPPED


def test_legacy_statuses_normalize_at_boundary():
    assert normalize_status('ok') == COMPLETED
    assert normalize_status('error') == FAILED
    assert normalize_status('cancelled') == STOPPED
    assert normalize_status('completed_with_warnings') == PARTIAL_SUCCESS
    assert normalize_status('cancelling') == 'cancelling'
    assert normalize_status('unknown') == FAILED


def test_richer_statuses_fit_the_v20_task_state_constraint():
    assert normalize_persisted_task_status('queued') == 'running'
    assert normalize_persisted_task_status('starting') == 'running'
    assert normalize_persisted_task_status('cancelling') == 'running'
    assert normalize_persisted_task_status('partial_success') == 'completed'
