import json
import os
import pandas as pd
import uuid
from flask import Blueprint, current_app, jsonify, request, send_file
from tabulate import tabulate
from typing import Any, Dict, List

from evalscope.config import TaskConfig
from evalscope.constants import EvalBackend, EvalType
from evalscope.report.combinator import get_data_frame, get_report_list
from evalscope.utils.logger import get_logger
from ..model_launcher import LaunchResult, LocalBackend, ModelSource, is_direct_eval_type, launch
from ..model_launcher import stop as launcher_stop
from ..utils import (
    DEFAULT_MULTIMODAL_BENCHMARKS,
    DEFAULT_TEXT_BENCHMARKS,
    OUTPUT_DIR,
    build_benchmark_entry,
    count_running_tasks,
    create_log_file,
    discover_all_benchmarks,
    get_log_content,
    run_eval_wrapper,
    run_in_subprocess,
    serialize_result,
    stop_process,
    try_reserve_slot,
    unregister_process,
    validate_task_id,
)

logger = get_logger()

bp_eval = Blueprint('eval', __name__, url_prefix='/api/v1/eval')


def _parse_mteb_results(work_dir: str) -> List:
    """Parse MTEB JSON results from results/ directory.

    MTEB format:
    {
      "task_name": "TaskName",
      "scores": {
        "test": [{"main_score": 0.5, ...}]
      }
    }

    Returns a list of objects with model_name, dataset_name, score, num.
    """
    from types import SimpleNamespace
    results_dir = os.path.join(work_dir, 'results')
    reports = []

    if not os.path.isdir(results_dir):
        return reports

    # Try to get sample count from task config or MTEB JSON files
    num_samples = _read_mteb_sample_count(work_dir)

    # Find all JSON files (excluding model_meta.json)
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith('.json') or fname == 'model_meta.json':
                continue

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                task_name = data.get('task_name', fname.replace('.json', ''))
                scores = data.get('scores', {})

                # Extract model name from path: results/eval__model_name/master/...
                rel_path = os.path.relpath(fpath, results_dir)
                parts = rel_path.split(os.sep)
                model_name = parts[0].replace('eval__', '') if parts else 'unknown'

                # Get main_score from first split (usually 'test')
                main_score = None
                for split_data in scores.values():
                    if isinstance(split_data, list) and split_data:
                        main_score = split_data[0].get('main_score')
                        break

                if main_score is not None:
                    reports.append(
                        SimpleNamespace(
                            model_name=model_name,
                            dataset_name=task_name,
                            score=main_score,
                            num=num_samples,
                        )
                    )
            except Exception as e:
                logger.warning(f'Failed to parse MTEB result {fpath}: {e}')

    return reports


def _read_mteb_sample_count(work_dir: str) -> int:
    """Read sample count from MTEB task config."""
    import yaml as _yaml
    config_path = os.path.join(work_dir, 'configs', 'task_config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path) as cf:
                cfg = _yaml.safe_load(cf) or {}
            limits = cfg.get('eval_config', {}).get('eval', {}).get('limits')
            if limits is not None:
                return int(limits)
        except Exception:
            pass
    return 0


_COLUMN_ZH = {
    'Model': '模型',
    'Dataset': '数据集',
    'Metric': '指标',
    'Subset': '子集',
    'Num': '数量',
    'Score': '得分',
}


def _build_result_table(work_dir: str) -> str:
    """Build a Markdown pipe-table from the JSON report files in *work_dir*/reports.

    Returns an empty string when no reports are found or on any error.
    """
    try:
        reports_dir = os.path.join(work_dir, 'reports')
        report_list = get_report_list([reports_dir])
        if not report_list:
            return ''
        df = get_data_frame(report_list, flatten_metrics=True, flatten_categories=True)
        _CAT_LEVEL_NAMES = ['类别', '子类别', '细分类别']
        new_cols = {}
        for col in df.columns:
            if col in _COLUMN_ZH:
                new_cols[col] = _COLUMN_ZH[col]
            elif col.startswith('Cat.'):
                try:
                    level = int(col[4:])
                    new_cols[col] = _CAT_LEVEL_NAMES[level] if level < len(_CAT_LEVEL_NAMES) else f'类别{level}'
                except ValueError:
                    new_cols[col] = col.replace('Cat.', '类别')
        df = df.rename(columns=new_cols)
        score_col = _COLUMN_ZH.get('Score', 'Score')
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col],
                                          errors='coerce').map(lambda x: f'{x:.4f}' if pd.notna(x) else '')
        return tabulate(df, headers=df.columns, tablefmt='pipe', showindex=False, disable_numparse=True)
    except Exception as e:
        logger.warning(f'Failed to build result table: {e}')
        return ''


_BASE_FIELDS = ['model', 'datasets']


def _get_required_fields(data: dict) -> list[str]:
    """Return required fields based on model_source. RAG eval has its own config."""
    if data.get('eval_backend') == EvalBackend.RAG_EVAL:
        return []  # RAG eval validates via eval_config, not top-level fields
    fields = list(_BASE_FIELDS)
    if data.get('model_source') == ModelSource.LOCAL:
        fields.append('model_path')
    else:
        fields.append('api_url')
    return fields


class RequestValidationError(Exception):
    """Raised by _parse_request when the incoming request is invalid."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@bp_eval.errorhandler(RequestValidationError)
def _handle_validation_error(exc: RequestValidationError):
    return jsonify({'error': exc.message}), exc.status_code


def _parse_request() -> tuple[dict, str]:
    """Validate the request body and return (data, task_id)."""
    data = request.get_json()
    if not data:
        raise RequestValidationError('Request body is required')

    for field in _get_required_fields(data):
        if field not in data:
            raise RequestValidationError(f'{field} is required')

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        raise RequestValidationError('EvalScope-Task-Id header is required')

    try:
        validate_task_id(task_id)
    except ValueError as e:
        raise RequestValidationError(str(e)) from e

    return data, task_id


def _build_task_config_openai(data: dict) -> TaskConfig:
    """Build a TaskConfig for OpenAI or Anthropic API mode."""
    api_eval_type = data.get('eval_type') or EvalType.OPENAI_API
    if not data.get('eval_type'):
        data['eval_type'] = api_eval_type
    # Auto-fill judge model args from eval model config when not provided
    if not data.get('judge_model_args'):
        data['judge_model_args'] = {
            'model_id': data.get('model'),
            'api_url': data.get('api_url'),
            'api_key': data.get('api_key'),
            'eval_type': api_eval_type,
        }
    task_config = TaskConfig.from_dict(data)
    task_config.no_timestamp = True
    task_config.enable_progress_tracker = True
    task_config.analysis_report = True
    return task_config


def _build_task_config_local(data: dict, launch_result: LaunchResult) -> TaskConfig:
    """Build a TaskConfig for local model mode."""
    task_config = TaskConfig(
        model=data.get('model') or os.path.basename(launch_result.model_path or data['model_path']),
        datasets=data.get('datasets', []),
        limit=data.get('limit'),
        eval_batch_size=data.get('eval_batch_size', 1),
        dataset_hub=data.get('dataset_hub', 'modelscope'),
        dataset_dir=data.get('dataset_dir') or '',
        dataset_args=data.get('dataset_args') or {},
        generation_config=data.get('generation_config', {}),
        repeats=data.get('repeats', 1),
        stream=data.get('stream'),
        no_timestamp=True,
        enable_progress_tracker=True,
        analysis_report=True,
        judge_model_args=data.get('judge_model_args') or {},
    )
    task_config.eval_type = launch_result.eval_type
    if launch_result.api_url:
        task_config.api_url = launch_result.api_url
        task_config.api_key = data.get('api_key') or launch_result.api_key
        # Clear checkpoint defaults (revision, precision) auto-set by TaskConfig
        # when eval_type was temporarily CHECKPOINT during __init__.
        task_config.model_args = {}
    else:
        task_config.model_args = launch_result.model_args
    return task_config


def _all_results_empty(result) -> bool:
    """Return True when every dataset in the evaluation result produced no scores.

    This happens when ``ignore_errors=True`` and every sample failed: each
    dataset evaluator returns an empty dict instead of a :class:`Report`.
    """
    if not result:
        return True
    if isinstance(result, dict):
        return all(not v for v in result.values())
    if isinstance(result, list):
        return all(_all_results_empty(r) for r in result)
    return False


def _execute_task(task_id: str, task_config: TaskConfig, label: str = 'Task', use_direct: bool = False):
    """Run the evaluation and return a Flask response."""
    create_log_file(task_id, os.path.join('logs', 'eval_log.log'))
    try:
        if use_direct:
            result = run_eval_wrapper(task_config)
        else:
            result = run_in_subprocess(
                run_eval_wrapper, task_config, task_id=task_id, task_type='eval', model=task_config.model
            )
        table_str = _build_result_table(task_config.work_dir)
        # RAG eval saves results in 'results/' not 'reports/'
        if task_config.eval_backend != EvalBackend.RAG_EVAL and _all_results_empty(result):
            error_msg = (
                'Evaluation completed but no results were produced. '
                'All samples may have failed. '
                'Check the evaluation log for details.'
            )
            logger.error(f'[{task_id}] {label} produced empty results: {error_msg}')
            return jsonify({'status': 'error', 'task_id': task_id, 'error': error_msg}), 500
        logger.info(f'[{task_id}] {label} completed successfully')

        # Write to SQLite (with retry for WAL lock contention from subprocess)
        try:
            import time as _time
            from datetime import datetime

            from .. import db as _db

            if task_config.eval_backend == EvalBackend.RAG_EVAL:
                # MTEB results are in results/ with MTEB JSON format
                report_list = _parse_mteb_results(task_config.work_dir)
            else:
                from evalscope.report.combinator import get_report_list
                reports_dir = os.path.join(task_config.work_dir, 'reports')
                report_list = get_report_list([reports_dir])
            if report_list:
                first = report_list[0]
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
                # Retry up to 3 times on lock, with backoff
                last_err = None
                for attempt in range(3):
                    try:
                        _db.upsert_eval_report(
                            task_id=task_id,
                            model_name=first.model_name,
                            dataset_name=', '.join(dataset_names) if len(dataset_names) > 1 else
                            (dataset_names[0] if dataset_names else ''),
                            score=avg_score,
                            num_samples=total_num,
                            timestamp=datetime.now().isoformat(),
                            dataset_scores=dataset_scores,
                        )
                        break
                    except Exception as e:
                        last_err = e
                        if 'locked' not in str(e).lower() or attempt == 2:
                            raise
                        _time.sleep(1 + attempt * 2)
                else:
                    raise last_err  # type: ignore[misc]
        except Exception as e:
            logger.warning(f'Failed to write eval to SQLite (non-fatal): {e}')

        return jsonify({
            'status': 'completed',
            'task_id': task_id,
            'result': serialize_result(result),
            'table': table_str
        })
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] [{task_id}] {label} failed: {e}', exc_info=True)
        return jsonify({'status': 'error', 'task_id': task_id, 'error': 'Task failed', 'error_id': error_id}), 500


@bp_eval.route('/invoke', methods=['POST'])
def run_evaluation():
    """Run a model evaluation task (blocking).

    Two modes:

    **OpenAI API** (model_source='openai' or not set)::

        {"model_source": "openai", "model": "qwen2.5-0.5b",
         "api_url": "http://localhost:8000/v1", "api_key": "sk-...",
         "datasets": ["gsm8k"], "limit": 10}

    **Local model** (model_source='local')::

        {"model_source": "local",
         "model_path": "/data/models/qwen.gguf", "backend": "auto",
         "backend_args": {"n_ctx": 2048}, "datasets": ["gsm8k"], "limit": 10}
    """
    # --- Concurrency guard (atomic check + reserve) ---
    data, task_id = _parse_request()
    model = data.get('model', '')
    if not try_reserve_slot(task_id, 'eval', model=model):
        max_eval = int(os.environ.get('MAX_CONCURRENT_EVAL', '2'))
        running = count_running_tasks('eval')
        return jsonify({
            'error': f'已有 {running} 个评估任务运行中，最大并发 {max_eval}，请等待完成后再试',
            'running': running,
            'max': max_eval,
        }), 429

    # Slot reserved – must clean up placeholder on any early return
    # before the subprocess is registered (which replaces it).
    try:
        model_source = data.get('model_source')

        # ── Local model: launch inference backend ──────────────────────
        launch_result: LaunchResult | None = None
        if model_source == ModelSource.LOCAL:
            model_path = data['model_path']
            backend = data.get('backend', LocalBackend.AUTO)
            backend_args = data.get('backend_args', {})
            logger.info(f'[{task_id}] Launching local model: path={model_path} backend={backend}')
            try:
                launch_result = launch(model_path, backend=backend, backend_args=backend_args)
                logger.info(
                    f'[{task_id}] Launched: backend={launch_result.backend} '
                    f'eval_type={launch_result.eval_type}'
                )
            except Exception as e:
                error_id = uuid.uuid4().hex[:8]
                logger.error(f'[{error_id}] [{task_id}] Launch failed: {e}', exc_info=True)
                return jsonify({
                    'status': 'error',
                    'task_id': task_id,
                    'error': 'Model launch failed',
                    'error_id': error_id
                }), 500

        # ── RAG Eval mode ────────────────────────────────────────────
        if data.get('eval_backend') == EvalBackend.RAG_EVAL:
            eval_config = data.get('eval_config', {})
            if not eval_config:
                return jsonify({'error': 'eval_config is required for RAG eval'}), 400

            tool = eval_config.get('tool', 'mteb')
            if tool == 'mteb':
                try:
                    import mteb  # noqa: F401
                except ImportError:
                    return jsonify({'error': 'MTEB is not installed. Please install: pip install "mteb>=2.7.0,<3.0.0"'}
                                   ), 400
            elif tool == 'ragas':
                try:
                    import ragas  # noqa: F401
                except ImportError:
                    return jsonify({
                        'error': 'RAGAS is not installed. Please install: pip install "ragas>=0.4.0,<0.5.0"'
                    }), 400
            elif tool == 'clip_benchmark':
                try:
                    import webdataset  # noqa: F401
                except ImportError:
                    return jsonify({'error': 'webdataset is not installed. Please install: pip install webdataset'}
                                   ), 400
            task_config = TaskConfig(
                eval_backend=EvalBackend.RAG_EVAL,
                eval_config=eval_config,
                work_dir=os.path.join(OUTPUT_DIR, task_id),
                no_timestamp=True,
                enable_progress_tracker=True,
            )
            task_config.work_dir = os.path.join(OUTPUT_DIR, task_id)
            os.makedirs(task_config.work_dir, exist_ok=True)
            try:
                task_config.dump_yaml(task_config.work_dir)
            except Exception as e:
                logger.warning(f'[{task_id}] Failed to save task config: {e}')
            logger.info(f'[{task_id}] Running RAG eval: tool={eval_config.get("tool")}')
            return _execute_task(task_id, task_config, label='RAG Eval')

        # ── Build TaskConfig ───────────────────────────────────────────
        try:
            if launch_result is not None:
                task_config = _build_task_config_local(data, launch_result)
            else:
                task_config = _build_task_config_openai(data)
        except Exception as e:
            if launch_result:
                launcher_stop(launch_result)
            error_id = uuid.uuid4().hex[:8]
            logger.error(f'[{error_id}] [{task_id}] Config build failed: {e}', exc_info=True)
            return jsonify({
                'status': 'error',
                'task_id': task_id,
                'error': 'Invalid configuration',
                'error_id': error_id
            }), 400

        task_config.work_dir = os.path.join(OUTPUT_DIR, task_id)

        # Save task config for resume capability
        os.makedirs(task_config.work_dir, exist_ok=True)
        try:
            task_config.dump_yaml(task_config.work_dir)
        except Exception as e:
            logger.warning(f'[{task_id}] Failed to save task config: {e}')

        logger.info(
            f'[{task_id}] Running: model={task_config.model} '
            f'eval_type={task_config.eval_type} datasets={task_config.datasets}'
        )

        # ── Execute ────────────────────────────────────────────────────
        try:
            use_direct = (
                launch_result is not None
                and (is_direct_eval_type(task_config.eval_type or '') or launch_result.api_url is not None)
            )
            return _execute_task(task_id, task_config, label='Task', use_direct=use_direct)
        finally:
            if launch_result:
                launcher_stop(launch_result)
    finally:
        # Clean up the placeholder if the subprocess was never registered
        # (i.e. we returned early before _execute_task / run_in_subprocess).
        unregister_process(task_id)


@bp_eval.route('/resume/invoke', methods=['POST'])
def resume_evaluation():
    """Resume an interrupted evaluation task (blocking).

    Request body::

        {"task_id": "eval_1782000000000"}

    This endpoint:
    1. Loads the original task_config.yaml from the task's work_dir
    2. Sets use_cache to the work_dir (enables prediction cache reuse)
    3. Runs the evaluation, skipping already-completed samples
    4. Returns the same response format as /invoke
    """
    data = request.get_json()
    if not data or 'task_id' not in data:
        return jsonify({'error': 'task_id is required'}), 400

    task_id = data['task_id']
    api_key = data.get('api_key')  # API key must be re-provided (not saved in config for security)
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    work_dir = os.path.join(OUTPUT_DIR, task_id)
    config_file = os.path.join(work_dir, 'task_config.yaml')
    progress_file = os.path.join(work_dir, 'progress.json')

    # Check if task exists
    if not os.path.exists(config_file):
        return jsonify({'error': f'Task not found: {task_id}'}), 404

    # Check if task is actually interrupted (status should be "running" but process is dead)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            status = progress.get('status')
            if status == 'completed':
                return jsonify({'error': 'Task already completed', 'task_id': task_id}), 400
            if status == 'error':
                # Allow resuming errored tasks
                pass
            # status == 'running' means interrupted (process died without cleanup)
        except Exception as e:
            logger.warning(f'Failed to read progress for {task_id}: {e}')

    # Concurrency guard
    model = ''  # We'll extract this from the config after loading
    if not try_reserve_slot(task_id, 'eval', model=model):
        max_eval = int(os.environ.get('MAX_CONCURRENT_EVAL', '2'))
        running = count_running_tasks('eval')
        return jsonify({
            'error': f'已有 {running} 个评估任务运行中，最大并发 {max_eval}，请等待完成后再试',
            'running': running,
            'max': max_eval,
        }), 429

    try:
        # Load original config
        try:
            task_config = TaskConfig.from_yaml(config_file)
        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f'[{error_id}] [{task_id}] Failed to load config: {e}', exc_info=True)
            return jsonify({
                'status': 'error',
                'task_id': task_id,
                'error': 'Failed to load task config',
                'error_id': error_id
            }), 500

        # Enable cache reuse - point to the same work_dir
        task_config.use_cache = work_dir
        task_config.work_dir = work_dir

        # Re-inject API key (stripped from saved config for security)
        if api_key:
            task_config.api_key = api_key

        logger.info(
            f'[{task_id}] Resuming: model={task_config.model} '
            f'datasets={task_config.datasets} use_cache={work_dir}'
        )

        # Execute (reuse _execute_task which handles subprocess + SQLite)
        return _execute_task(task_id, task_config, label='Resume', use_direct=False)
    finally:
        unregister_process(task_id)


@bp_eval.route('/stop', methods=['POST'])
def stop_evaluation():
    """Stop a running evaluation task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    stopped = stop_process(task_id)
    if stopped:
        return jsonify({'status': 'stopped', 'task_id': task_id}), 200
    else:
        return jsonify({'error': f'No running task found for task_id: {task_id}'}), 404


@bp_eval.route('/progress', methods=['GET'])
def get_evaluation_progress():
    """Get the real-time hierarchical progress of a running evaluation task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    progress_file = os.path.join(OUTPUT_DIR, task_id, 'progress.json')
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return jsonify(progress), 200
    except FileNotFoundError:
        # If the progress file doesn't exist AND the task directory itself
        # doesn't exist (or was deleted), the task is not a valid running task
        if not os.path.isdir(task_dir):
            return jsonify({'error': f'Task not found: {task_id}'}), 404
        return jsonify({'percent': 0.0}), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get progress for task {task_id}: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get progress', 'error_id': error_id}), 500


@bp_eval.route('/progress/stream', methods=['GET'])
def stream_evaluation_progress():
    """SSE stream for real-time evaluation progress updates.

    Query params:
        task_id (str): the task identifier

    Returns a text/event-stream that pushes progress JSON whenever the
    progress file changes.  The stream closes when the task reaches 100%
    or the client disconnects.
    """
    import time

    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    progress_file = os.path.join(OUTPUT_DIR, task_id, 'progress.json')

    def generate():
        last_mtime = 0
        idle_count = 0
        max_idle = 300  # Close after 5 minutes of no progress updates
        while True:
            try:
                if os.path.isfile(progress_file):
                    mtime = os.path.getmtime(progress_file)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        idle_count = 0
                        with open(progress_file, 'r') as f:
                            data = json.load(f)
                        yield f'data: {json.dumps(data)}\n\n'
                        if data.get('percent', 0) >= 100 and data.get('status') == 'completed':
                            break
                    else:
                        idle_count += 1
                        if idle_count >= max_idle:
                            yield f'data: {json.dumps({"event": "timeout", "message": "SSE idle timeout"})}\n\n'
                            break
                else:
                    idle_count += 1
                    if idle_count >= max_idle:
                        yield f'data: {json.dumps({"event": "timeout", "message": "SSE idle timeout"})}\n\n'
                        break
                # Send heartbeat every 30s to keep connection alive
                if idle_count % 30 == 0 and idle_count > 0:
                    yield f': heartbeat\n\n'
                time.sleep(1)
            except GeneratorExit:
                break
            except Exception as e:
                logger.debug(f'SSE progress stream error for {task_id}: {e}')
                time.sleep(2)

    from flask import Response
    return Response(generate(), mimetype='text/event-stream')


@bp_eval.route('/report', methods=['GET'])
def get_evaluation_report():
    """Get the HTML evaluation report for a completed task.

    Query params:
        task_id (str): the task identifier
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    report_file = os.path.join(OUTPUT_DIR, task_id, 'reports', 'report.html')
    if not os.path.exists(report_file):
        return jsonify({'error': f'Report not found for task_id: {task_id}'}), 404

    return send_file(report_file, mimetype='text/html')


@bp_eval.route('/log', methods=['GET'])
def get_evaluation_log():
    """Get evaluation log content with pagination.

    Query params:
        task_id    (str): the task identifier
        start_line (int, optional): if not provided, read last `page` lines from end
        page       (int): number of lines to read (default 500)

    Returns:
        dict with text, head_line, tail_line, total_lines
    """
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    start_line = request.args.get('start_line', type=int)
    page = request.args.get('page', 500, type=int)

    try:
        result = get_log_content(task_id, os.path.join('logs', 'eval_log.log'), start_line, page)
        return jsonify(result), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get evaluation log: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get log', 'error_id': error_id}), 500


@bp_eval.route('/log/stream', methods=['GET'])
def stream_evaluation_log():
    """SSE stream for real-time evaluation log updates.

    Query params:
        task_id (str): the task identifier

    Pushes new log lines as they are appended to the log file.
    The stream closes when the client disconnects.
    """
    import time

    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400
    # Support resume: client can pass last_pos (byte offset) to skip already-seen content
    try:
        initial_pos = int(request.args.get('last_pos', 0))
    except (ValueError, TypeError):
        initial_pos = 0

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    log_file = os.path.join(OUTPUT_DIR, task_id, 'logs', 'eval_log.log')

    def generate():
        # Send initial heartbeat to confirm connection is alive
        yield ': connected\n\n'
        last_pos = initial_pos
        idle_count = 0
        max_idle = 300  # Close after 5 minutes of no new log lines
        while True:
            try:
                if os.path.isfile(log_file):
                    with open(log_file, 'r') as f:
                        f.seek(last_pos)
                        new_content = f.read()
                        if new_content:
                            last_pos = f.tell()
                            idle_count = 0
                            # Send as SSE event with position for resume support
                            payload = json.dumps({'text': new_content, 'pos': last_pos})
                            yield f'data: {payload}\n\n'
                        else:
                            idle_count += 1
                            if idle_count >= max_idle:
                                yield f'data: {json.dumps({"event": "timeout", "message": "SSE idle timeout"})}\n\n'
                                break
                else:
                    idle_count += 1
                    if idle_count >= max_idle:
                        yield f'data: {json.dumps({"event": "timeout", "message": "SSE idle timeout"})}\n\n'
                        break
                # Send heartbeat every 30s
                if idle_count % 30 == 0 and idle_count > 0:
                    yield f': heartbeat\n\n'
                time.sleep(1)
            except GeneratorExit:
                break
            except Exception as e:
                logger.debug(f'SSE log stream error for {task_id}: {e}')
                time.sleep(2)

    from flask import Response
    return Response(generate(), mimetype='text/event-stream')


@bp_eval.route('/benchmarks', methods=['GET'])
def list_benchmarks():
    """Return the catalogue of supported benchmarks with descriptions.

    The list is split into two categories: ``text`` (LLM-only) and
    ``multimodal`` (VLM).  Descriptions are loaded from the ``_meta`` JSON
    files and post-processed: the H1 title and the last H2 section are
    stripped, then the remainder is split into per-section blocks.

    The default catalogue can be overridden at application startup by setting
    ``app.config['SUPPORTED_BENCHMARKS']`` to a dict with keys ``'text'`` and
    ``'multimodal'``, each containing a list of benchmark names.

    Query params:
        type (str, optional): Filter to ``'text'`` or ``'multimodal'`` only.
        all (str, optional): When ``'true'``, return *all* benchmarks discovered
            from the ``_meta`` directory instead of the curated default lists.
    """
    try:
        filter_type = request.args.get('type', '').lower()
        return_all = request.args.get('all', '').lower() == 'true'

        if return_all:
            # Discover every benchmark from _meta directory
            all_names = discover_all_benchmarks()
            all_entries = [build_benchmark_entry(name) for name in all_names]

            if filter_type == 'text':
                result = {'text': [e for e in all_entries if e.get('category') == 'llm']}
            elif filter_type == 'multimodal':
                result = {'multimodal': [e for e in all_entries if e.get('category') == 'vlm']}
            else:
                result = {
                    'text': [e for e in all_entries if e.get('category') == 'llm'],
                    'multimodal': [e for e in all_entries if e.get('category') == 'vlm'],
                }
        else:
            # Use the curated default lists (backward-compatible)
            cfg = current_app.config.get('SUPPORTED_BENCHMARKS', {})
            text_names: List[str] = cfg.get('text', DEFAULT_TEXT_BENCHMARKS)
            multimodal_names: List[str] = cfg.get('multimodal', DEFAULT_MULTIMODAL_BENCHMARKS)

            result: Dict[str, Any] = {}
            if filter_type in ('', 'text'):
                result['text'] = [build_benchmark_entry(name) for name in text_names]
            if filter_type in ('', 'multimodal'):
                result['multimodal'] = [build_benchmark_entry(name) for name in multimodal_names]

        if filter_type and filter_type not in ('text', 'multimodal'):
            return jsonify({'error': f"Unknown type '{filter_type}'. Use 'text' or 'multimodal'."}), 400

        return jsonify(result), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to list benchmarks: {e}', exc_info=True)
        return jsonify({'error': 'Failed to list benchmarks', 'error_id': error_id}), 500
