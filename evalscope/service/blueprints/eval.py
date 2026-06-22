import json
import os
import pandas as pd
import uuid
from flask import Blueprint, current_app, jsonify, request, send_file
from tabulate import tabulate
from typing import Any, Dict, List

from evalscope.config import TaskConfig
from evalscope.constants import EvalType
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
    """Return required fields based on model_source."""
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
    """Build a TaskConfig for OpenAI API mode."""
    if not data.get('eval_type'):
        data['eval_type'] = EvalType.OPENAI_API
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
        if _all_results_empty(result):
            error_msg = (
                'Evaluation completed but no results were produced. '
                'All samples may have failed. '
                'Check the evaluation log for details.'
            )
            logger.error(f'[{task_id}] {label} produced empty results: {error_msg}')
            return jsonify({'status': 'error', 'task_id': task_id, 'error': error_msg}), 500
        logger.info(f'[{task_id}] {label} completed successfully')

        # Write to SQLite
        try:
            from datetime import datetime

            from evalscope.report.combinator import get_report_list
            from .. import db as _db
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
        except Exception as e:
            logger.debug(f'Failed to write eval to SQLite (non-fatal): {e}')

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
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return jsonify(progress), 200
    except FileNotFoundError:
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
                        if data.get('percent', 0) >= 100:
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

    log_file = os.path.join(OUTPUT_DIR, task_id, 'logs', 'eval_log.log')

    def generate():
        last_pos = 0
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
                            # Send as SSE event
                            payload = json.dumps({'text': new_content})
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
