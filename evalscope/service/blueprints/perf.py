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

    Query params:
        root_path (str): output root directory (default: OUTPUT_DIR)
        search    (str): search in model name and dataset
        model     (str): filter by model (exact match)
        dataset   (str): filter by dataset (exact match)
        sort_by   (str): sort field ('time', 'model')
        sort_order (str): 'asc' or 'desc' (default: 'desc')
    """
    root = request.args.get('root_path', OUTPUT_DIR)
    search = request.args.get('search', '').strip().lower()
    filter_model = request.args.get('model', '').strip()
    filter_dataset = request.args.get('dataset', '').strip()
    sort_by = request.args.get('sort_by', 'time')
    sort_order = request.args.get('sort_order', 'desc')

    try:
        root = validate_root_path(root)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    if not os.path.isdir(root):
        return jsonify({'tasks': [], 'root_path': root, 'error': f'Directory not found: {root}'}), 200

    tasks = []
    all_models = set()
    all_datasets = set()

    for entry in sorted(os.listdir(root), reverse=True):
        task_dir = os.path.join(root, entry)
        perf_dir = os.path.join(task_dir, 'perf')
        if not os.path.isdir(task_dir) or not os.path.isdir(perf_dir):
            continue

        # Default metadata
        meta = {
            'task_id': entry,
            'model': 'N/A',
            'api': 'N/A',
            'dataset': 'N/A',
            'runs': 0,
            'has_report': os.path.exists(os.path.join(perf_dir, 'perf_report.html')),
            'timestamp': '',
        }

        # Try to read args from first run subdirectory (look in task_dir and perf_dir)
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
        except Exception:
            pass

        # Count run subdirs (look in both task_dir and perf_dir)
        try:
            run_count = 0
            for search_dir in [task_dir, perf_dir]:
                if os.path.isdir(search_dir):
                    run_count += sum(
                        1 for s in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, s)) and s != 'perf'
                    )
            meta['runs'] = run_count
        except Exception:
            pass

        # Timestamp from mtime
        try:
            mtime = os.path.getmtime(task_dir)
            from datetime import datetime
            meta['timestamp'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

        tasks.append(meta)
        if meta['model'] != 'N/A':
            all_models.add(meta['model'])
        if meta['dataset'] != 'N/A':
            all_datasets.add(meta['dataset'])

    # Apply filters
    if search:
        tasks = [t for t in tasks if search in t['model'].lower() or search in t['dataset'].lower()]
    if filter_model:
        tasks = [t for t in tasks if t['model'] == filter_model]
    if filter_dataset:
        tasks = [t for t in tasks if t['dataset'] == filter_dataset]

    # Sort
    if sort_by == 'model':
        tasks.sort(key=lambda t: t['model'].lower(), reverse=(sort_order == 'desc'))
    else:
        # Default: sort by time (already in reverse order from listdir)
        if sort_order == 'asc':
            tasks.reverse()

    return jsonify({
        'tasks': tasks,
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
    # --- Concurrency guard ---
    max_perf = int(os.environ.get('MAX_CONCURRENT_PERF', '1'))
    running = count_running_tasks('perf')
    if running >= max_perf:
        return jsonify({
            'error': f'已有 {running} 个压测任务运行中，最大并发 {max_perf}，请等待完成后再试',
            'running': running,
            'max': max_perf,
        }), 429

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400

    # url is required for remote APIs, but local models auto-generate their own URL
    api_type = data.get('api', 'openai')
    required_fields = ['model']
    if not api_type.startswith('local'):
        required_fields.append('url')
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    task_id = request.headers.get('EvalScope-Task-Id')
    if not task_id:
        return jsonify({'error': 'EvalScope-Task-Id header is required'}), 400
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Default to openai API
    if 'api' not in data:
        data['api'] = 'openai'

    perf_args = PerfArguments.from_dict(data)
    perf_args.no_timestamp = True
    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
    perf_args.name = 'perf'
    perf_args.enable_progress_tracker = True
    perf_args.no_test_connection = True

    logger.info(f'[{task_id}] Running performance benchmark for model: {perf_args.model}')
    logger.info(f'[{task_id}] URL: {perf_args.url}')

    create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

    try:
        result = run_in_subprocess(
            run_perf_wrapper, perf_args, task_id=task_id, task_type='perf', model=perf_args.model
        )
        table_str = _build_perf_table(result, api_type=perf_args.api)
        logger.info(f'[{task_id}] Task completed successfully')
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
