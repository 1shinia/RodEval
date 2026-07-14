import contextlib
import io
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Active process registry – allows external stop by task_id
# ---------------------------------------------------------------------------


@dataclass
class TaskInfo:
    """Metadata for a running task."""

    task_id: str
    task_type: str  # 'eval' or 'perf'
    model: str = ''
    start_time: float = field(default_factory=time.time)
    process: multiprocessing.Process | None = None


_active_processes: dict[str, TaskInfo] = {}
"""Maps task_id → TaskInfo for the currently running subprocess."""

_active_lock = threading.Lock()


def register_process(
    task_id: str,
    proc: multiprocessing.Process,
    task_type: str = '',
    model: str = '',
) -> None:
    """Register a running subprocess so it can be stopped later."""
    with _active_lock:
        _active_processes[task_id] = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            model=model,
            process=proc,
        )


def get_running_tasks() -> list[dict]:
    """Return metadata for all currently running tasks.
    
    A task is considered running if it exists in _active_processes.
    We trust unregister_process to remove tasks when they complete,
    rather than checking is_alive() which has race conditions.
    """
    with _active_lock:
        result = []
        for info in _active_processes.values():
            result.append({
                'task_id': info.task_id,
                'task_type': info.task_type,
                'model': info.model,
                'start_time': info.start_time,
                'elapsed_seconds': round(time.time() - info.start_time, 1),
            })
        return result


def count_running_tasks(task_type: str | None = None) -> int:
    """Return the number of currently running tasks.

    If *task_type* is given (e.g. ``'eval'`` or ``'perf'``), only count
    tasks of that type.  Otherwise count all running tasks.

    Note: entries with ``process=None`` (placeholder slots reserved by
    :func:`try_reserve_slot`) are also counted so that the number
    reflects actual concurrency usage.
    """
    with _active_lock:
        return sum(
            1 for info in _active_processes.values()
            if task_type is None or info.task_type == task_type
        )


def try_reserve_slot(task_id: str, task_type: str, model: str = '') -> bool:
    """Atomically check the concurrency limit and reserve a slot.

    Returns ``True`` if the slot was reserved (count < max), ``False`` if
    the limit has been reached.  When ``True`` a placeholder *TaskInfo* is
    inserted into the registry so that subsequent calls see the increased
    count immediately — closing the race window between checking and
    registering.

    The caller MUST call :func:`finalize_slot` (or :func:`unregister_process`)
    to either attach the real process or clean up the placeholder.
    """
    max_key = f'MAX_CONCURRENT_{task_type.upper()}'
    max_slots = int(os.environ.get(max_key, '2' if task_type == 'eval' else '1'))
    with _active_lock:
        running = sum(
            1 for info in _active_processes.values()
            if info.task_type == task_type
        )
        if running >= max_slots:
            return False
        # Insert a placeholder so other threads see the slot as taken
        _active_processes[task_id] = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            model=model,
            process=None,  # will be filled in by finalize_slot()
        )
        return True


def finalize_slot(task_id: str, proc: multiprocessing.Process) -> None:
    """Attach the real subprocess to a previously reserved slot.

    If no placeholder exists (e.g. ``run_in_subprocess`` was called
    without a prior ``try_reserve_slot``), a fresh entry is created.
    """
    with _active_lock:
        info = _active_processes.get(task_id)
        if info is not None:
            info.process = proc
        else:
            # No placeholder — register fresh (backward compatible).
            _active_processes[task_id] = TaskInfo(
                task_id=task_id,
                task_type='',
                model='',
                process=proc,
            )
    # Persist to SQLite
    try:
        from .. import db as _db
        _db.upsert_task_state(
            task_id=task_id,
            task_type=info.task_type if info else '',
            status='running',
            pid=proc.pid,
            model=info.model if info else '',
        )
    except Exception as e:
        logger.debug(f'Failed to persist task state for {task_id}: {e}')


def unregister_process(task_id: str) -> None:
    """Remove a finished / stopped subprocess from the registry."""
    with _active_lock:
        _active_processes.pop(task_id, None)
    # Remove from SQLite (task is done, no longer "running")
    try:
        from .. import db as _db
        _db.delete_task_state(task_id)
    except Exception as e:
        logger.debug(f'Failed to clean task state for {task_id}: {e}')


def stop_process(task_id: str) -> bool:
    """Terminate the subprocess associated with *task_id*.

    Returns True if a process was found and terminated, False otherwise.
    Uses process group kill (os.killpg) to ensure all child/grandchild processes
    are also terminated, preventing orphaned GPU workers or model servers.
    """
    with _active_lock:
        info = _active_processes.pop(task_id, None)
    if info is None:
        return False
    # Allow stopping placeholder tasks (process not yet attached):
    # the subprocess will find no registry entry and exit cleanly.
    if info.process is None:
        logger.info(f'Task {task_id} (placeholder) stopped by user.')
        try:
            from .. import db as _db
            _db.delete_task_state(task_id)
        except Exception:
            pass
        return True
    proc = info.process
    if proc.is_alive() and proc.pid is not None:
        try:
            # Get process group ID (child calls os.setsid() on startup).
            # BUT: there is a race window between p.start() and os.setsid()
            # where the child is still in the parent's process group.
            # os.killpg() during that window would kill the Flask server!
            child_pgid = os.getpgid(proc.pid)
            parent_pgid = os.getpgid(os.getpid())

            if child_pgid == parent_pgid:
                # Child has not yet called os.setsid() — it's still in our
                # process group.  Use os.kill() to target only the child.
                logger.warning(
                    f'Task {task_id} is still in the parent process group '
                    '(os.setsid() not yet called). Using os.kill() instead of '
                    'os.killpg() to avoid killing the service.'
                )
                os.kill(proc.pid, signal.SIGTERM)
                proc.join(timeout=3)
                if proc.is_alive():
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.join(timeout=2)
            else:
                # Child is in its own process group — safe to use killpg
                # to clean up the entire process tree.
                os.killpg(child_pgid, signal.SIGTERM)
                proc.join(timeout=3)
                if proc.is_alive():
                    # Force kill the entire process group
                    os.killpg(child_pgid, signal.SIGKILL)
                    proc.join(timeout=2)
        except ProcessLookupError:
            # Process already exited, ignore
            pass
    logger.info(f'Task {task_id} stopped by user.')
    # Update SQLite so the UI reflects the stopped state immediately,
    # rather than waiting for run_in_subprocess to call unregister_process.
    try:
        from .. import db as _db
        _db.delete_task_state(task_id)
    except Exception as e:
        logger.debug(f'Failed to clean task state for {task_id}: {e}')
    return True


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_stderr():
    """Context manager that redirects sys.stderr to a StringIO buffer.

    Yields the buffer so the caller can read captured output after the block.
    Always restores the original sys.stderr on exit.
    """
    buf = io.StringIO()
    original = sys.stderr
    sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stderr = original


def _process_worker(func, result_queue, *args, **kwargs):
    """Target for multiprocessing.Process — executes *func* and posts result.

    stderr is captured and forwarded through the queue so the parent process
    can surface it even when the child crashes before *func* is reached.

    stdout is redirected to /dev/null to prevent BrokenPipeError from
    libraries (e.g. Rich, tqdm) that write to stdout when the parent's
    stdout pipe has been closed (common under web servers).
    """
    # Create a new session so this process becomes the process group leader.
    # This allows the parent to kill the entire process tree (this process +
    # any children it spawns) via os.killpg() when stopping the task.
    os.setsid()
    try:
        with _capture_stderr() as stderr_buf:
            with open(os.devnull, 'w') as devnull:
                original_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result = func(*args, **kwargs)
                    result_queue.put({'status': 'success', 'result': result})
                except BaseException as e:
                    result_queue.put({
                        'status': 'error',
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'stderr': stderr_buf.getvalue(),
                    })
                finally:
                    sys.stdout = original_stdout
    except BaseException:
        # Suppress cleanup-phase noise (e.g. asyncio event loop AbortError
        # during subprocess teardown) — the real result was already posted.
        pass


def run_in_subprocess(func, *args, task_id=None, task_type='', model='', **kwargs):
    """Run *func* in a child process and return its result (blocks caller).

    Returns the function's return value on success; raises on error.

    If *task_id* is provided the child process is registered in the active
    process registry so it can be terminated via :func:`stop_process`.

    Design note — why polling instead of p.join() then queue.get():
    ``multiprocessing.Queue`` is backed by an OS pipe whose buffer is typically
    only 64 KB.  If the child calls ``queue.put()`` with a payload larger than
    that buffer it will *block* until the parent drains the pipe.  But if the
    parent is sitting in ``p.join()`` waiting for the child to exit first, both
    sides wait on each other forever — a classic deadlock.
    """
    # Use spawn context to avoid fork-based deadlocks on Linux and to
    # ensure consistent cross-platform behaviour (macOS defaults to spawn,
    # Linux defaults to fork).
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    p = ctx.Process(target=_process_worker, args=(func, result_queue, *args), kwargs=kwargs)
    p.start()

    if task_id:
        # If a placeholder was reserved by try_reserve_slot, attach the
        # real process to it; otherwise register a fresh entry.
        finalize_slot(task_id, p)

    res = None
    # Poll for the result while the child is alive so we continuously drain
    # the underlying pipe and never let queue.put() block in the child.
    while p.is_alive():
        try:
            res = result_queue.get(timeout=0.1)
            break  # Got the result; let the child finish normally.
        except queue.Empty:
            continue  # Child still running — keep draining.

    # Wait for the child to clean up after we have the result (or it crashed).
    p.join()

    if task_id:
        unregister_process(task_id)

    if res is not None:
        if res['status'] == 'error':
            stderr_info = res.get('stderr', '')
            stderr_section = f'\n[stderr]\n{stderr_info}' if stderr_info.strip() else ''
            raise RuntimeError(f"Subprocess error: {res['error']}\n{res.get('traceback', '')}{stderr_section}")
        return res['result']

    # res is still None: the child exited without putting anything in the queue
    # (OOM, SIGKILL, import error, segfault, etc.).
    # Do one final non-blocking check in case the item arrived between the last
    # loop iteration and p.join() returning.
    try:
        res = result_queue.get_nowait()
        if res['status'] == 'error':
            stderr_info = res.get('stderr', '')
            stderr_section = f'\n[stderr]\n{stderr_info}' if stderr_info.strip() else ''
            raise RuntimeError(f"Subprocess error: {res['error']}\n{res.get('traceback', '')}{stderr_section}")
        return res['result']
    except queue.Empty:
        pass

    raise RuntimeError(
        f'Subprocess terminated unexpectedly (exit code {p.exitcode}). '
        'The child process may have crashed due to OOM, a missing import, '
        'GPU initialisation failure, or a signal (e.g. SIGKILL).'
    ) from None


# ---------------------------------------------------------------------------
# Task wrappers (thin shims kept for clarity / future extension)
# ---------------------------------------------------------------------------


def run_eval_wrapper(task_config: TaskConfig):
    """Run an evaluation task and return the result."""
    result = run_task(task_config)
    _persist_eval_report(task_config)
    return result


def _persist_eval_report(task_config: TaskConfig) -> None:
    """Write the evaluation report from disk into the SQLite metadata DB."""
    try:
        import time as _time
        from datetime import datetime

        from evalscope.report.combinator import get_report_list
        from evalscope.service.db import init_db, upsert_eval_report

        # init_db must be called in the subprocess (global state is per-process)
        output_dir = os.path.dirname(task_config.work_dir.rstrip('/'))
        init_db(output_dir)

        reports_dir = os.path.join(task_config.work_dir, 'reports')
        if not os.path.isdir(reports_dir):
            # RAG eval (MTEB) saves results in 'results/' instead of 'reports/'
            _persist_rag_results(task_config)
            return
        report_list = get_report_list([reports_dir])
        if not report_list:
            return
        first = report_list[0]
        task_id = os.path.basename(task_config.work_dir.rstrip('/'))
        total_num = sum(r.num or 0 for r in report_list)
        dataset_names = [r.dataset_name for r in report_list]
        score_sum = sum(r.score for r in report_list if r.score is not None)
        avg_score = round(score_sum / len(report_list), 4) if report_list else 0.0
        dataset_scores = {}
        for r in report_list:
            score = r.score
            if score is not None and score > 1:
                score = score / 100
            dataset_scores[r.dataset_name] = round(score, 4) if score is not None else None
        for attempt in range(3):
            try:
                upsert_eval_report(
                    task_id=task_id,
                    model_name=first.model_name,
                    dataset_name=', '.join(dataset_names) if len(dataset_names) > 1 else
                    (dataset_names[0] if dataset_names else ''),
                    score=avg_score,
                    num_samples=total_num,
                    timestamp=datetime.now().isoformat(),
                    dataset_scores=dataset_scores,
                )
                return
            except Exception as e:
                if 'locked' not in str(e).lower() or attempt == 2:
                    raise
                _time.sleep(1 + attempt * 2)
    except Exception as e:
        from evalscope.utils.logger import get_logger
        get_logger().warning(f'Failed to persist eval report to SQLite (non-fatal): {e}')


def _persist_rag_results(task_config: TaskConfig) -> None:
    """Persist MTEB/RAG evaluation results from the ``results/`` directory."""
    import glob
    import json
    from datetime import datetime

    from evalscope.service.db import upsert_eval_report

    results_dir = os.path.join(task_config.work_dir, 'results')
    if not os.path.isdir(results_dir):
        return

    task_id = os.path.basename(task_config.work_dir.rstrip('/'))
    result_files = glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True)
    dataset_files = [f for f in result_files if not f.endswith('model_meta.json') and 'run_settings' not in f]

    if not dataset_files:
        return

    model_name = ''
    scores: dict = {}
    task_names: set = set()
    total_samples = 0
    # Extract model name from directory: results/<eval__model_name>/...
    import re
    for rf in dataset_files:
        m = re.search(r'/results/(?:eval__)?([^/]+)/', rf)
        if m:
            model_name = m.group(1)
            break
    # Try to get sample count from task config limits, fall back to experiment count
    try:
        import yaml
        config_path = os.path.join(task_config.work_dir, 'configs', 'task_config.yaml')
        if os.path.exists(config_path):
            with open(config_path) as cf:
                cfg = yaml.safe_load(cf) or {}
            eval_cfg = cfg.get('eval_config', {})
            if isinstance(eval_cfg, dict):
                total_samples = eval_cfg.get('eval', {}).get('limits', 0)
    except Exception:
        total_samples = 0
    for rf in dataset_files:
        try:
            with open(rf) as fh:
                data = json.load(fh)
            task_name = data.get('task_name', os.path.basename(rf).replace('.json', ''))
            task_names.add(task_name)
            # Count experiments as fallback sample count
            for split_data in data.get('scores', {}).values():
                if isinstance(split_data, list):
                    for exp in split_data:
                        sc = exp.get('scores_per_experiment', [])
                        if sc and not total_samples:
                            total_samples = max(total_samples, len(sc))
            task_score = data.get('scores', {}).get('test', [{}])[0].get('main_score')
            if task_score is None:
                # fallback: compute mean of all experiment scores
                all_scores = []
                for split_data in data.get('scores', {}).values():
                    for exp in split_data:
                        for metric_v in exp.get('scores_per_experiment', []):
                            for k, v in metric_v.items():
                                if k == data.get('mteb_version', '')[:0] or not isinstance(v, (int, float)):
                                    continue
                                all_scores.append(v)
                task_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
            task_name = data.get('task_name', os.path.basename(rf).replace('.json', ''))
            scores[task_name] = round(float(task_score), 4) if task_score is not None else None
        except Exception:
            continue

    if not scores:
        return

    avg = round(sum(v for v in scores.values() if v is not None) / len(scores), 4)
    dataset_str = ', '.join(scores.keys())
    try:
        upsert_eval_report(
            task_id=task_id,
            model_name=model_name or 'unknown',
            dataset_name=dataset_str,
            score=avg,
            num_samples=total_samples or len(task_names) or 10,
            timestamp=datetime.now().isoformat(),
            dataset_scores=scores,
        )
    except Exception as e:
        from evalscope.utils.logger import get_logger
        get_logger().warning(f'Failed to persist RAG results to SQLite (non-fatal): {e}')


def run_perf_wrapper(perf_args: PerfArguments):
    """Run a performance benchmark and return the result."""
    return run_perf_benchmark(perf_args)


def serialize_result(result):
    """Convert Pydantic model objects (or containers of them) to plain dicts for JSON.

    Recursively walks dicts and lists, converting any Pydantic ``BaseModel``
    instances (``Report``, ``BenchmarkSummary``, ``PercentileResult``, etc.)
    to plain dicts via ``model_dump()`` / ``to_dict()``.
    Also sanitises NaN / Infinity to None (JSON-compatible).
    """
    from math import isfinite
    from pydantic import BaseModel

    if isinstance(result, float) and not isfinite(result):
        return None
    if isinstance(result, BaseModel):
        # Report has a custom to_dict() that delegates to model_dump()
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        return result.model_dump()
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [serialize_result(v) for v in result]
    return result
