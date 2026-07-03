import json
import os
import uuid
from flask import Blueprint, jsonify, request, send_file
from tabulate import tabulate

from evalscope.perf.arguments import Arguments as PerfArguments
from evalscope.perf.utils.benchmark_util import Metrics
from evalscope.perf.utils.rich_display import EmbeddingResultAnalyzer, LLMResultAnalyzer
from evalscope.utils.logger import get_logger
from ..utils import (
    OUTPUT_DIR,
    count_running_tasks,
    create_log_file,
    get_log_content,
    run_in_subprocess,
    run_perf_wrapper,
    serialize_result,
    stop_process,
    try_reserve_slot,
    unregister_process,
    validate_root_path,
    validate_task_id,
)

logger = get_logger()


def _build_perf_table(result, api_type: str = None) -> str:
    """Build a Markdown pipe-table from perf benchmark results with Chinese headers.

    Returns an empty string when no valid results are found.
    """
    try:
        is_emb = Metrics.is_embedding_or_rerank(api_type)
        analyzer = EmbeddingResultAnalyzer() if is_emb else LLMResultAnalyzer()
        analysis = analyzer.analyze(result)
        if not analysis.rows:
            return ''
        if is_emb:
            headers = ['并发数', '请求速率', '每秒请求数', '平均延迟(s)', 'P99延迟(s)', '平均输入TPS', 'P99输入TPS', '平均输入Token数', '成功率']
        else:
            headers = [
                '并发数', '请求速率', '请求数', '每秒请求数', '平均延迟(s)', 'P99延迟(s)', '平均首字延迟(s)', 'P99首字延迟(s)', '平均每Token延迟(s)',
                'P99每Token延迟(s)', '生成速度(toks/s)', '成功率'
            ]
        return tabulate([list(r.values()) for r in analysis.rows], headers=headers, tablefmt='pipe')
    except Exception as e:
        logger.warning(f'Failed to build perf table: {e}')
        return ''


bp_perf = Blueprint('perf', __name__, url_prefix='/api/v1/perf')


@bp_perf.route('/list', methods=['GET'])
def list_perf_tasks():
    """List all performance test tasks with metadata.

    Uses SQLite for fast queries.  Falls back to filesystem scan if the
    DB is not initialised.

    Query params:
        root_path  (str): output root directory (default: OUTPUT_DIR)
        search     (str): search in model name and dataset
        model      (str): filter by model (exact match)
        dataset    (str): filter by dataset (exact match)
        sort_by    (str): sort field ('time', 'model')
        sort_order (str): 'asc' or 'desc' (default: 'desc')
        page       (int): page number (default: 1)
        page_size  (int): items per page (default: 20)
    """
    root = request.args.get('root_path', OUTPUT_DIR)
    search = request.args.get('search', '').strip().lower()
    filter_model = request.args.get('model', '').strip()
    filter_dataset = request.args.get('dataset', '').strip()
    sort_by = request.args.get('sort_by', 'time')
    sort_order = request.args.get('sort_order', 'desc')
    page = max(1, request.args.get('page', 1, type=int))
    page_size = max(1, min(100, request.args.get('page_size', 20, type=int)))

    try:
        root = validate_root_path(root)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if not os.path.isdir(root):
        return jsonify({'tasks': [], 'root_path': root, 'error': f'Directory not found: {root}'}), 200

    # --- Try SQLite first ---
    try:
        from .. import db as _db
        items, total, available_models, available_datasets = _db.query_perf_tasks(
            search=search,
            filter_model=filter_model,
            filter_dataset=filter_dataset,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )
        return jsonify({
            'tasks': items,
            'total': total,
            'page': page,
            'page_size': page_size,
            'root_path': root,
            'filters': {
                'available_models': available_models,
                'available_datasets': available_datasets,
            },
        }), 200
    except Exception as db_err:
        logger.debug(f'SQLite query failed, falling back to filesystem: {db_err}')

    # --- Fallback: filesystem scan (original logic) ---
    tasks = []
    all_models = set()
    all_datasets = set()

    for entry in sorted(os.listdir(root), reverse=True):
        task_dir = os.path.join(root, entry)
        perf_dir = os.path.join(task_dir, 'perf')
        if not os.path.isdir(task_dir) or not os.path.isdir(perf_dir):
            continue

        meta = {
            'task_id': entry,
            'model': 'N/A',
            'api': 'N/A',
            'dataset': 'N/A',
            'runs': 0,
            'has_report': os.path.exists(os.path.join(perf_dir, 'perf_report.html')),
            'timestamp': '',
        }

        try:
            search_dirs = [task_dir, perf_dir]
            for search_dir in search_dirs:
                if not os.path.isdir(search_dir):
                    continue
                for sub in sorted(os.listdir(search_dir)):
                    sub_dir = os.path.join(search_dir, sub)
                    if not os.path.isdir(sub_dir) or sub == 'perf':
                        continue
                    args_file = os.path.join(sub_dir, 'benchmark_args.json')
                    if os.path.isfile(args_file):
                        with open(args_file, 'r') as f:
                            args_data = json.load(f)
                        meta['model'] = args_data.get('model', 'N/A')
                        meta['api'] = args_data.get('api', 'N/A')
                        meta['dataset'] = args_data.get('dataset_label') or args_data.get('dataset', 'N/A')
                        break
                if meta['model'] != 'N/A':
                    break
        except Exception as e:
            logger.debug(f'Failed to read args for task {entry}: {e}')

        try:
            run_count = 0
            for search_dir in [task_dir, perf_dir]:
                if os.path.isdir(search_dir):
                    run_count += sum(
                        1 for s in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, s)) and s != 'perf'
                    )
            meta['runs'] = run_count
        except Exception as e:
            logger.debug(f'Failed to count runs for task {entry}: {e}')

        try:
            mtime = os.path.getmtime(task_dir)
            from datetime import datetime
            meta['timestamp'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.debug(f'Failed to get timestamp for task {entry}: {e}')

        tasks.append(meta)
        if meta['model'] != 'N/A':
            all_models.add(meta['model'])
        if meta['dataset'] != 'N/A':
            all_datasets.add(meta['dataset'])

    if search:
        tasks = [t for t in tasks if search in t['model'].lower() or search in t['dataset'].lower()]
    if filter_model:
        tasks = [t for t in tasks if t['model'] == filter_model]
    if filter_dataset:
        tasks = [t for t in tasks if t['dataset'] == filter_dataset]

    if sort_by == 'model':
        tasks.sort(key=lambda t: t['model'].lower(), reverse=(sort_order == 'desc'))
    else:
        if sort_order == 'asc':
            tasks.reverse()

    total = len(tasks)
    start = (page - 1) * page_size
    page_tasks = tasks[start:start + page_size]

    return jsonify({
        'tasks': page_tasks,
        'total': total,
        'page': page,
        'page_size': page_size,
        'root_path': root,
        'filters': {
            'available_models': sorted(m for m in all_models if m),
            'available_datasets': sorted(d for d in all_datasets if d),
        },
    }), 200


@bp_perf.route('/invoke', methods=['POST'])
def run_performance_test():
    """Run a performance benchmark task (blocking).

    Returns the benchmark result when the task completes.
    """
    # --- Parse task_id first (needed for slot reservation) ---
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        return jsonify({'error': 'EvalScope-Task-Id header is required'}), 400
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # --- Concurrency guard (atomic check + reserve) ---
    model = data.get('model', '')
    if not try_reserve_slot(task_id, 'perf', model=model):
        max_perf = int(os.environ.get('MAX_CONCURRENT_PERF', '1'))
        running = count_running_tasks('perf')
        return jsonify({
            'error': f'已有 {running} 个压测任务运行中，最大并发 {max_perf}，请等待完成后再试',
            'running': running,
            'max': max_perf,
        }), 429

    try:
        # url is required for remote APIs, but local models auto-generate their own URL
        api_type = data.get('api', 'openai')
        required_fields = ['model']
        if not api_type.startswith('local'):
            required_fields.append('url')
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400

        # Default to openai API
        if 'api' not in data:
            data['api'] = 'openai'

        perf_args = PerfArguments.from_dict(data)
        perf_args.no_timestamp = True
        perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
        perf_args.name = 'perf'
        perf_args.enable_progress_tracker = True
        perf_args.no_test_connection = True

        # Save task config for resume capability (strip api_key for security)
        os.makedirs(perf_args.outputs_dir, exist_ok=True)
        try:
            save_data = {k: v for k, v in data.items() if k != 'api_key'}
            config_file = os.path.join(perf_args.outputs_dir, 'task_config.json')
            with open(config_file, 'w') as f:
                json.dump(save_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f'[{task_id}] Failed to save task config: {e}')

        logger.info(f'[{task_id}] Running performance benchmark for model: {perf_args.model}')
        logger.info(f'[{task_id}] URL: {perf_args.url}')

        create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

        try:
            result = run_in_subprocess(
                run_perf_wrapper, perf_args, task_id=task_id, task_type='perf', model=perf_args.model
            )
            table_str = _build_perf_table(result, api_type=perf_args.api)
            logger.info(f'[{task_id}] Task completed successfully')

            # Write to SQLite
            try:
                from datetime import datetime

                from .. import db as _db
                perf_dir = os.path.join(OUTPUT_DIR, task_id, 'perf')
                has_report = os.path.exists(os.path.join(perf_dir, 'perf_report.html'))
                runs = 0
                for search_dir in [os.path.join(OUTPUT_DIR, task_id), perf_dir]:
                    if os.path.isdir(search_dir):
                        runs += sum(
                            1 for s in os.listdir(search_dir)
                            if os.path.isdir(os.path.join(search_dir, s)) and s != 'perf'
                        )
                _db.upsert_perf_task(
                    task_id=task_id,
                    model=perf_args.model,
                    api=perf_args.api,
                    dataset=perf_args.dataset_label or perf_args.dataset or 'N/A',
                    runs=runs,
                    has_report=has_report,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                )
            except Exception as e:
                logger.debug(f'Failed to write perf to SQLite (non-fatal): {e}')

            return jsonify({
                'status': 'completed',
                'task_id': task_id,
                'result': serialize_result(result),
                'table': table_str
            })
        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f'[{error_id}] [{task_id}] Task failed: {e}', exc_info=True)
            return jsonify({'status': 'error', 'task_id': task_id, 'error': 'Task failed', 'error_id': error_id}), 500
    finally:
        # Clean up the placeholder if the subprocess was never registered
        unregister_process(task_id)


@bp_perf.route('/stop', methods=['POST'])
def stop_performance_test():
    """Stop a running performance benchmark task.

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


@bp_perf.route('/resume/invoke', methods=['POST'])
def resume_performance_test():
    """Resume an interrupted performance benchmark task (blocking).

    Request body::

        {"task_id": "perf_1782000000000", "api_key": "sk-..."}

    This endpoint:
    1. Loads the original task_config.json from the task's work_dir
    2. Reconstructs PerfArguments from the saved config
    3. Runs the benchmark, reusing the same output directory
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
    config_file = os.path.join(work_dir, 'task_config.json')

    # Check if task exists
    if not os.path.exists(config_file):
        return jsonify({'error': f'Task not found or not resumable: {task_id}'}), 404

    # Concurrency guard
    model = data.get('model', '')
    if not try_reserve_slot(task_id, 'perf', model=model):
        max_perf = int(os.environ.get('MAX_CONCURRENT_PERF', '1'))
        running = count_running_tasks('perf')
        return jsonify({
            'error': f'已有 {running} 个压测任务运行中，最大并发 {max_perf}，请等待完成后再试',
            'running': running,
            'max': max_perf,
        }), 429

    try:
        # Load original config
        try:
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f'[{error_id}] [{task_id}] Failed to load config: {e}', exc_info=True)
            return jsonify({
                'status': 'error',
                'task_id': task_id,
                'error': 'Failed to load task config',
                'error_id': error_id
            }), 500

        # Re-inject API key (stripped from saved config for security)
        if api_key:
            saved_data['api_key'] = api_key

        # Build PerfArguments from saved config
        perf_args = PerfArguments.from_dict(saved_data)
        perf_args.no_timestamp = True
        perf_args.outputs_dir = work_dir
        perf_args.name = 'perf'
        perf_args.enable_progress_tracker = True
        perf_args.no_test_connection = True

        logger.info(
            f'[{task_id}] Resuming: model={perf_args.model} '
            f'api={saved_data.get("api", "openai")} outputs_dir={work_dir}'
        )

        # Re-create log file (appends to existing)
        create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

        # Clean up old benchmark databases from previous interrupted runs
        # (perf library refuses to overwrite existing .db files)
        import glob
        for old_db in glob.glob(os.path.join(work_dir, 'perf', '**', 'benchmark_data.db'), recursive=True):
            try:
                os.remove(old_db)
                logger.info(f'[{task_id}] Removed old database: {old_db}')
            except Exception as e:
                logger.warning(f'[{task_id}] Failed to remove {old_db}: {e}')

        try:
            result = run_in_subprocess(
                run_perf_wrapper, perf_args, task_id=task_id, task_type='perf', model=perf_args.model
            )
            table_str = _build_perf_table(result, api_type=perf_args.api)
            logger.info(f'[{task_id}] Task completed successfully')

            # Update SQLite
            try:
                from datetime import datetime

                from .. import db as _db
                perf_dir = os.path.join(OUTPUT_DIR, task_id, 'perf')
                has_report = os.path.exists(os.path.join(perf_dir, 'perf_report.html'))
                runs = 0
                for search_dir in [os.path.join(OUTPUT_DIR, task_id), perf_dir]:
                    if os.path.isdir(search_dir):
                        runs += sum(
                            1 for s in os.listdir(search_dir)
                            if os.path.isdir(os.path.join(search_dir, s)) and s != 'perf'
                        )
                _db.upsert_perf_task(
                    task_id=task_id,
                    model=perf_args.model,
                    api=perf_args.api,
                    dataset=perf_args.dataset_label or perf_args.dataset or 'N/A',
                    runs=runs,
                    has_report=has_report,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                )
            except Exception as e:
                logger.debug(f'Failed to write perf to SQLite (non-fatal): {e}')

            return jsonify({
                'status': 'completed',
                'task_id': task_id,
                'result': serialize_result(result),
                'table': table_str
            })
        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f'[{error_id}] [{task_id}] Task failed: {e}', exc_info=True)
            return jsonify({'status': 'error', 'task_id': task_id, 'error': 'Task failed', 'error_id': error_id}), 500
    finally:
        unregister_process(task_id)


@bp_perf.route('/delete', methods=['DELETE'])
def delete_performance_test():
    """Delete a performance test task directory.

    JSON body:
        task_id (str): the task identifier
    """
    data = request.get_json()
    if not data or not data.get('task_id'):
        return jsonify({'error': 'task_id is required'}), 400

    task_id = data['task_id']
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    import shutil
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    if not os.path.isdir(task_dir):
        return jsonify({'error': f'Task not found: {task_id}'}), 404

    try:
        shutil.rmtree(task_dir)
        logger.info(f'Deleted perf task: {task_id}')

        # Sync SQLite
        try:
            from .. import db as _db
            _db.delete_perf_task(task_id)
        except Exception as e:
            logger.debug(f'Failed to delete from SQLite (non-fatal): {e}')

        return jsonify({'ok': True, 'task_id': task_id}), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to delete perf task {task_id}: {e}', exc_info=True)
        return jsonify({'error': 'Failed to delete task', 'error_id': error_id}), 500


@bp_perf.route('/report', methods=['GET'])
def get_performance_report():
    """Get the HTML performance report for a completed task.

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

    report_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'perf_report.html')
    if not os.path.exists(report_file):
        return jsonify({'error': f'Report not found for task_id: {task_id}'}), 404

    return send_file(report_file, mimetype='text/html')


@bp_perf.route('/log', methods=['GET'])
def get_performance_log():
    """Get performance benchmark log content with pagination.

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
        result = get_log_content(task_id, os.path.join('perf', 'benchmark.log'), start_line, page)
        return jsonify(result), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get performance log: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get log', 'error_id': error_id}), 500


@bp_perf.route('/log/stream', methods=['GET'])
def stream_performance_log():
    """SSE stream for real-time performance log updates.

    Query params:
        task_id (str): the task identifier

    Pushes new log lines as they are appended to the log file.
    The stream closes when the client disconnects.
    """
    import time

    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'task_id is required'}), 400

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    log_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'benchmark.log')

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


@bp_perf.route('/progress', methods=['GET'])
def get_performance_progress():
    """Get the real-time hierarchical progress of a running perf benchmark task.

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

    progress_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'progress.json')
    task_dir = os.path.join(OUTPUT_DIR, task_id)
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return jsonify(progress), 200
    except FileNotFoundError:
        # If the progress file doesn't exist AND the task directory itself
        # doesn't exist (or was deleted), the task is not a valid running task
        # — return 404 so the frontend treats it as completed/not-found.
        if not os.path.isdir(task_dir):
            return jsonify({'error': f'Task not found: {task_id}'}), 404
        # progress.json missing but task_dir exists — task may be starting up
        return jsonify({'percent': 0.0}), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get progress for task {task_id}: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get progress', 'error_id': error_id}), 500


@bp_perf.route('/progress/stream', methods=['GET'])
def stream_performance_progress():
    """SSE stream for real-time performance progress updates.

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

    progress_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'progress.json')

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
