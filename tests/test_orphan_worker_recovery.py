"""Service-restart recovery must fence off workers from the dead service."""

import multiprocessing
import os
import signal
import time

import pytest

from evalscope.service import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    db._db_path = None
    db._local.conn = None
    db.init_db(str(tmp_path))
    yield
    db._db_path = None
    db._local.conn = None


def _seed_running_task(task_id: str, pid: int, pid_start_ticks: int | None = None) -> None:
    db.upsert_task_state(
        task_id=task_id,
        task_type='eval',
        status='running',
        pid=pid,
        model='model-x',
        user_id=1,
        pid_start_ticks=pid_start_ticks,
    )


def _spawn_style_worker(ready):
    os.setsid()
    ready.set()
    signal.pause()


def test_live_spawn_worker_identity_is_recognized_and_terminated():
    ctx = multiprocessing.get_context('spawn')
    ready = ctx.Event()
    worker = ctx.Process(target=_spawn_style_worker, args=(ready,))
    worker.start()
    try:
        assert ready.wait(20), f'spawn worker did not initialize, exitcode={worker.exitcode}'
        deadline = time.monotonic() + 5
        start_ticks = db._process_start_ticks(worker.pid)
        assert start_ticks is not None
        while time.monotonic() < deadline and not db._is_confirmed_stale_worker(worker.pid, start_ticks):
            time.sleep(0.05)
        assert db._is_confirmed_stale_worker(worker.pid, start_ticks)
        assert db._terminate_stale_worker(worker.pid, start_ticks, timeout=1)
        worker.join(timeout=2)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            os.kill(worker.pid, signal.SIGKILL)
            worker.join(timeout=2)


def test_recovery_terminates_confirmed_old_worker_before_marking_orphaned(monkeypatch):
    _seed_running_task('eval_old_worker', 4242)
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    terminated = []

    def terminate(pid, start_ticks):
        terminated.append((pid, start_ticks))
        return True

    monkeypatch.setattr(db, '_terminate_stale_worker', terminate, raising=False)
    db._get_conn().execute(
        "UPDATE task_state SET pid_start_ticks = 99 WHERE task_id = 'eval_old_worker'"
    )
    db._get_conn().commit()

    assert db.recover_stale_tasks() == ['eval_old_worker']
    assert terminated == [(4242, 99)]
    state = db.get_all_task_states()[0]
    assert state['status'] == 'orphaned'


def test_recovery_fails_closed_when_live_pid_cannot_be_safely_terminated(monkeypatch):
    _seed_running_task('eval_unverified_worker', 4343)
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    monkeypatch.setattr(db, '_terminate_stale_worker', lambda pid, ticks: False, raising=False)

    with pytest.raises(db.StaleWorkerRecoveryError, match='eval_unverified_worker'):
        db.recover_stale_tasks()
    state = db.get_all_task_states()[0]
    assert state['status'] == 'running'


def test_reused_pid_with_different_start_time_is_never_signalled(monkeypatch):
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    monkeypatch.setattr(db, '_process_start_ticks', lambda pid: 200)
    kill_calls = []
    monkeypatch.setattr(db.os, 'killpg', lambda pid, sig: kill_calls.append((pid, sig)))

    assert not db._terminate_stale_worker(4444, 100, timeout=0)
    assert kill_calls == []


def test_sweep_keeps_live_legacy_worker_without_identity(monkeypatch):
    _seed_running_task('eval_legacy_worker', 4545)
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)

    alive = db.sweep_orphaned_tasks()

    assert [row['task_id'] for row in alive] == ['eval_legacy_worker']
    assert db.get_all_task_states()[0]['status'] == 'running'


def test_sweep_orphans_reused_pid_with_mismatched_identity(monkeypatch):
    _seed_running_task('eval_reused_pid', 4646, pid_start_ticks=100)
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    monkeypatch.setattr(db, '_process_start_ticks', lambda pid: 200)

    assert db.sweep_orphaned_tasks() == []
    assert db.get_all_task_states()[0]['status'] == 'orphaned'


def test_matching_service_process_identity_skips_recovery(tmp_path, monkeypatch):
    _seed_running_task('eval_live_service', 4747, pid_start_ticks=100)
    pid_file = tmp_path / 'evalscope_service.pid'
    pid_file.write_text('5151 300')
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    monkeypatch.setattr(db, '_process_start_ticks', lambda pid: 300)
    terminated = []
    monkeypatch.setattr(
        db, '_terminate_stale_worker',
        lambda pid, ticks: terminated.append((pid, ticks)) or True,
    )

    assert db.recover_stale_tasks() == []
    assert terminated == []
    assert db.get_all_task_states()[0]['status'] == 'running'


def test_reused_service_pid_does_not_skip_worker_fencing(tmp_path, monkeypatch):
    _seed_running_task('eval_reused_service_pid', 4848, pid_start_ticks=100)
    pid_file = tmp_path / 'evalscope_service.pid'
    pid_file.write_text('5151 300')
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    monkeypatch.setattr(db, '_process_start_ticks', lambda pid: 301 if pid == 5151 else 100)
    terminated = []
    monkeypatch.setattr(
        db, '_terminate_stale_worker',
        lambda pid, ticks: terminated.append((pid, ticks)) or True,
    )

    assert db.recover_stale_tasks() == ['eval_reused_service_pid']
    assert terminated == [(4848, 100)]
    assert db.get_all_task_states()[0]['status'] == 'orphaned'


def test_live_legacy_service_pid_fails_closed(tmp_path, monkeypatch):
    _seed_running_task('eval_legacy_service', 4898, pid_start_ticks=100)
    (tmp_path / 'evalscope_service.pid').write_text('5151')
    monkeypatch.setattr(db, '_pid_alive', lambda pid: True)
    terminated = []
    monkeypatch.setattr(
        db, '_terminate_stale_worker',
        lambda pid, ticks: terminated.append((pid, ticks)) or True,
    )

    with pytest.raises(db.StaleWorkerRecoveryError, match='legacy service PID 5151'):
        db.recover_stale_tasks()
    assert terminated == []
    assert db.get_all_task_states()[0]['status'] == 'running'


def test_dead_leader_with_live_group_member_is_not_fenced(monkeypatch):
    monkeypatch.setattr(db, '_pid_alive', lambda pid: False)
    monkeypatch.setattr(db, '_process_group_members', lambda pgid: {4949: 200})

    assert not db._terminate_stale_worker(4948, 100, timeout=0)


def test_service_pid_file_persists_process_identity(tmp_path):
    db.write_service_pid(str(tmp_path))

    assert db._read_service_pid(str(tmp_path / 'evalscope_service.pid')) == (
        os.getpid(), db._process_start_ticks(os.getpid()),
    )
