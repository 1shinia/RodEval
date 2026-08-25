import json
import os
import uuid
from flask import Blueprint, current_app, jsonify, request, send_file
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

# ═══════════════════════════════════════════════════════════════════════════════
# Batch model testing — template download / CSV upload / batch run
# ═══════════════════════════════════════════════════════════════════════════════

BATCH_CSV_TEMPLATE = os.path.join(os.path.dirname(OUTPUT_DIR), 'data', 'model_list.csv')
BATCH_UPLOAD_DIR = os.path.join(OUTPUT_DIR, '_batch_uploads')


def _mark_perf_completed(task_id: str) -> None:
    """Write a ``completed`` progress.json marker for a finished perf task.

    Perf task dirs do not otherwise write progress.json, so the startup
    retention cleanup (log.py ``cleanup_old_task_logs``) treats every perf
    dir as incomplete and deletes it after the 7-day retention window.
    This marker aligns perf dirs with completed eval dirs, which are kept
    forever. Called after a perf task finishes successfully.
    """
    from datetime import datetime

    try:
        pj = os.path.join(OUTPUT_DIR, task_id, 'progress.json')
        os.makedirs(os.path.dirname(pj), exist_ok=True)
        with open(pj, 'w', encoding='utf-8') as f:
            json.dump({
                'status': 'completed',
                'phase': 'completed',
                'pipeline': 'perf',
                'updated_at': datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f'[{task_id}] Failed to write perf completion marker: {e}')


@bp_perf.route('/template', methods=['GET'])
def download_template():
    """Download the model list CSV template."""
    if not os.path.isfile(BATCH_CSV_TEMPLATE):
        return jsonify({'error': 'Template file not found on server'}), 404
    return send_file(
        BATCH_CSV_TEMPLATE,
        mimetype='text/csv',
        as_attachment=True,
        download_name='model_list_template.csv',
    )


@bp_perf.route('/batch/upload', methods=['POST'])
def upload_batch_csv():
    """Upload a model list CSV for batch testing.

    Returns parsed preview: model count + first few rows.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files are accepted'}), 400

    import csv as csv_mod
    content = f.read().decode('utf-8-sig')
    reader = csv_mod.DictReader(content.splitlines())

    rows = []
    for row in reader:
        if not row.get('model'):
            continue
        enabled = row.get('enabled', 'TRUE').strip().upper()
        if enabled == 'FALSE':
            continue
        rows.append({
            'name': row['model'].strip(),
            'base_url': (row.get('base_url') or '').strip(),
            'api_key': (row.get('api_key') or '').strip(),
            'api': (row.get('api') or 'openai').strip(),
            'model': row['model'].strip(),
            'concurrency': (row.get('concurrency') or '').strip(),
            'number': (row.get('number') or '').strip(),
            'max_tokens': int(row['max_tokens']) if row.get('max_tokens', '').strip() else 0,
            'stream': (row.get('stream', 'TRUE') or 'TRUE').strip().upper() == 'TRUE',
            'prompt': (row.get('prompt') or '').strip(),
        })

    if not rows:
        return jsonify({'error': 'CSV 中没有有效的模型行（name/model 为空或被 disabled）'}), 400

    # Save to temp file on server for later batch run
    os.makedirs(BATCH_UPLOAD_DIR, exist_ok=True)
    batch_id = uuid.uuid4().hex[:12]
    saved_path = os.path.join(BATCH_UPLOAD_DIR, f'{batch_id}.csv')
    with open(saved_path, 'w', encoding='utf-8') as outf:
        outf.write(content)

    return jsonify({
        'batch_id': batch_id,
        'model_count': len(rows),
        'models': [r['name'] for r in rows],
        'preview': rows[:5],
    }), 200


@bp_perf.route('/batch/launch', methods=['POST'])
def launch_batch_perf():
    """Launch batch performance tests in background.

    JSON body: same as old /batch/run.
    Returns immediately with batch_id.  Use /batch/status/<batch_id> to poll.
    """
    import csv as csv_mod
    import threading
    from datetime import datetime
    from .auth import get_current_user_id

    data = request.get_json()
    if not data or not data.get('batch_id'):
        return jsonify({'error': 'batch_id is required'}), 400

    batch_id = data['batch_id']
    csv_path = os.path.join(BATCH_UPLOAD_DIR, f'{batch_id}.csv')
    if not os.path.isfile(csv_path):
        return jsonify({'error': f'Batch file not found: {batch_id}. Please re-upload.'}), 404

    if batch_id in _batch_state and _batch_state[batch_id].get('status') == 'running':
        return jsonify({'error': 'Batch already running'}), 409

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv_mod.DictReader(f)
        model_rows = list(reader)

    total = sum(1 for r in model_rows
                if (r.get('model') or '').strip()
                and (r.get('enabled', 'TRUE') or 'TRUE').strip().upper() != 'FALSE')

    state = {
        'batch_id': batch_id,
        'status': 'running',
        'total': total,
        'completed': 0,
        'errors': 0,
        'current_model': '',
        'results': [],
        'error_details': [],
        'cancel_requested': False,
    }
    _batch_state[batch_id] = state

    # Capture user_id for the background thread
    shared_config = {
        'user_id': get_current_user_id(),
        'parallel': data.get('parallel', [1]),
        'number': data.get('number', [10]),
        'rate': data.get('rate'),
        'max_tokens': data.get('max_tokens'),
        'min_tokens': data.get('min_tokens'),
        'dataset': data.get('dataset', 'openqa'),
        'dataset_path': data.get('dataset_path'),
        'max_prompt_length': data.get('max_prompt_length'),
        'min_prompt_length': data.get('min_prompt_length'),
        'prefix_length': data.get('prefix_length'),
        'tokenizer_path': data.get('tokenizer_path'),
        'extra_args': data.get('extra_args'),
        'warmup_num': data.get('warmup_num'),
        'duration': data.get('duration'),
    }

    def _run_batch():
        state = _batch_state.get(batch_id)
        if not state:
            return
        try:
            for row in model_rows:
                if state['cancel_requested']:
                    state['status'] = 'cancelled'
                    logger.info(f'[batch:{batch_id}] Cancelled by user')
                    break

                model_name = (row.get('model') or '').strip()
                if not model_name:
                    continue
                enabled = (row.get('enabled', 'TRUE') or 'TRUE').strip().upper()
                if enabled == 'FALSE':
                    continue

                state['current_model'] = model_name
                task_id = f'perf_{int(datetime.now().timestamp() * 1000)}'
                state['current_task_id'] = task_id

                # Parse CSV concurrency: comma-separated like "1,2,4"
                csv_concurrency = (row.get('concurrency') or '').strip()
                if csv_concurrency:
                    parallel = [int(x.strip()) for x in csv_concurrency.replace('，', ',').split(',') if x.strip()]
                else:
                    parallel = shared_config.get('parallel', [1])
                if not parallel:
                    parallel = [1]

                # Parse CSV number: comma-separated, fallback to shared_config
                csv_number = (row.get('number') or '').strip()
                if csv_number:
                    number = [int(x.strip()) for x in csv_number.replace('，', ',').split(',') if x.strip()]
                else:
                    number = shared_config.get('number', [10])
                if not number:
                    number = [10]

                # max_tokens: CSV overrides, fallback to shared_config
                csv_max_tokens = (row.get('max_tokens') or '').strip()
                if csv_max_tokens:
                    max_tok = int(csv_max_tokens)
                elif shared_config.get('max_tokens'):
                    max_tok = int(shared_config['max_tokens'])
                else:
                    max_tok = 200

                stream = (row.get('stream', 'TRUE') or 'TRUE').strip().upper() == 'TRUE'

                perf_data = {
                    'model': model_name,
                    'api': (row.get('api') or 'openai').strip(),
                    'url': (row.get('base_url') or '').strip(),
                    'api_key': (row.get('api_key') or '').strip(),
                    'parallel': parallel,
                    'number': number,
                    'max_tokens': max_tok,
                    'stream': stream,
                    'dataset': shared_config['dataset'],
                }
                if row.get('prompt', '').strip():
                    perf_data['prompt'] = row['prompt'].strip()
                if shared_config['rate']:
                    perf_data['rate'] = shared_config['rate']
                if shared_config['min_tokens']:
                    perf_data['min_tokens'] = shared_config['min_tokens']
                if shared_config['max_prompt_length']:
                    perf_data['max_prompt_length'] = shared_config['max_prompt_length']
                if shared_config['min_prompt_length']:
                    perf_data['min_prompt_length'] = shared_config['min_prompt_length']
                if shared_config['prefix_length']:
                    perf_data['prefix_length'] = shared_config['prefix_length']
                if shared_config['tokenizer_path']:
                    perf_data['tokenizer_path'] = shared_config['tokenizer_path']
                if shared_config['dataset_path']:
                    perf_data['dataset_path'] = shared_config['dataset_path']
                if shared_config['extra_args']:
                    perf_data['extra_args'] = shared_config['extra_args']
                if shared_config.get('read_timeout'):
                    perf_data['read_timeout'] = shared_config['read_timeout']
                if shared_config.get('warmup_num'):
                    perf_data['warmup_num'] = shared_config['warmup_num']
                if shared_config.get('duration'):
                    perf_data['duration'] = shared_config['duration']

                if not try_reserve_slot(task_id, 'perf', model=model_name, user_id=shared_config['user_id']):
                    state['errors'] += 1
                    state['error_details'].append({'name': model_name, 'model': model_name, 'error': '并发已满'})
                    continue

                logger.info(f'[batch:{batch_id}] Running perf for {model_name}')

                try:
                    perf_args = PerfArguments.from_dict(perf_data)
                    perf_args.no_timestamp = True
                    perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
                    perf_args.name = 'perf'
                    perf_args.enable_progress_tracker = True
                    perf_args.no_test_connection = True
                    # Default read timeout: prevent hanging on stalled streams
                    if perf_args.read_timeout is None:
                        perf_args.read_timeout = 300

                    os.makedirs(perf_args.outputs_dir, exist_ok=True)
                    save_data = {k: v for k, v in perf_data.items() if k != 'api_key'}
                    config_file = os.path.join(perf_args.outputs_dir, 'task_config.json')
                    with open(config_file, 'w') as cf:
                        json.dump(save_data, cf, ensure_ascii=False)

                    create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

                    result = run_in_subprocess(
                        run_perf_wrapper, perf_args, task_id=task_id, task_type='perf', model=perf_args.model
                    )

                    # Check if the benchmark actually succeeded (not just "ran without crashing")
                    perf_success = True
                    perf_error_msg = ''
                    perf_dir = os.path.join(OUTPUT_DIR, task_id, 'perf')
                    try:
                        for entry in os.listdir(perf_dir):
                            summary_path = os.path.join(perf_dir, entry, 'benchmark_summary.json')
                            if not os.path.isfile(summary_path):
                                continue
                            with open(summary_path, 'r') as sf:
                                summary = json.load(sf)
                            failed = summary.get('Failed Requests', 0)
                            success = summary.get('Success Requests', 0)
                            if failed > 0 and success == 0:
                                perf_success = False
                                # Extract error from benchmark log
                                log_path = os.path.join(perf_dir, 'benchmark.log')
                                if os.path.isfile(log_path):
                                    with open(log_path, 'r') as lf:
                                        for line in lf:
                                            if 'Non-retryable error' in line or 'Invalid' in line or 'error' in line.lower():
                                                perf_error_msg = line.strip()
                                                break
                                if not perf_error_msg:
                                    perf_error_msg = f'所有 {failed} 个请求失败（Success: {summary.get("Success Requests", 0)}）'
                                break
                    except Exception:
                        pass

                    if not perf_success:
                        state['errors'] += 1
                        state['completed'] += 1
                        state['results'].append({
                            'task_id': task_id,
                            'name': model_name,
                            'model': model_name,
                            'status': 'error',
                            'error': perf_error_msg,
                        })
                        state['error_details'].append({
                            'name': model_name,
                            'model': model_name,
                            'error': perf_error_msg,
                        })
                        logger.warning(f'[batch:{batch_id}] [{task_id}] {model_name} 压测失败: {perf_error_msg}')
                    else:

                        # Write to SQLite
                        try:
                            from .. import db as _db
                            perf_dir = os.path.join(OUTPUT_DIR, task_id, 'perf')
                            has_report = os.path.exists(os.path.join(perf_dir, 'perf_report.html'))
                            runs = 0
                            for sd in [os.path.join(OUTPUT_DIR, task_id), perf_dir]:
                                if os.path.isdir(sd):
                                    runs += sum(1 for s in os.listdir(sd) if os.path.isdir(os.path.join(sd, s)) and s != 'perf')
                            _db.upsert_perf_task(
                                task_id=task_id,
                                model=perf_args.model,
                                api=perf_args.api,
                                dataset=perf_args.dataset_label or perf_args.dataset or 'N/A',
                                runs=runs,
                                has_report=has_report,
                                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                user_id=shared_config['user_id'],
                            )
                        except Exception as e:
                            logger.error(f'Failed to write perf to SQLite (data remains on disk, will backfill on restart): {e}')

                        state['completed'] += 1
                        state['results'].append({'task_id': task_id, 'name': model_name, 'model': model_name, 'status': 'completed'})
                        logger.info(f'[batch:{batch_id}] [{task_id}] {model_name} completed ({state["completed"]}/{total})')

                except Exception as e:
                    error_id = uuid.uuid4().hex[:8]
                    logger.error(f'[batch:{batch_id}] [{task_id}] {model_name} failed: {e}', exc_info=True)
                    state['errors'] += 1
                    state['error_details'].append({'name': model_name, 'model': model_name, 'error': str(e)})
                finally:
                    unregister_process(task_id)
                    state['current_model'] = ''
                    state['current_task_id'] = ''

            if state['status'] == 'running':
                state['status'] = 'completed'
        except Exception as e:
            state['status'] = 'error'
            logger.error(f'[batch:{batch_id}] Fatal error: {e}', exc_info=True)

    thread = threading.Thread(target=_run_batch, daemon=True)
    thread.start()

    return jsonify({'batch_id': batch_id, 'total': total, 'status': 'launched'}), 200


@bp_perf.route('/batch/status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id: str):
    """Get the current status of a running batch test."""
    state = _batch_state.get(batch_id)
    if not state:
        return jsonify({'error': 'Batch not found'}), 404
    return jsonify({
        'batch_id': batch_id,
        'status': state['status'],
        'total': state['total'],
        'completed': state['completed'],
        'errors': state['errors'],
        'current_model': state['current_model'],
        'current_task_id': state.get('current_task_id', ''),
        'results': state.get('results', []),
        'error_details': state.get('error_details', []),
    }), 200


@bp_perf.route('/batch/stop/<batch_id>', methods=['POST'])
def stop_batch_perf(batch_id: str):
    """Request cancellation of a running batch test."""
    state = _batch_state.get(batch_id)
    if not state:
        return jsonify({'error': 'Batch not found'}), 404
    if state['status'] != 'running':
        return jsonify({'error': f'Batch is not running (status: {state["status"]})'}), 400
    state['cancel_requested'] = True

    # Kill the currently running subprocess for immediate stop
    current_task_id = state.get('current_task_id', '')
    if current_task_id:
        stop_process(current_task_id)

    return jsonify({'batch_id': batch_id, 'status': 'cancelling'}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# Batch state dict  (module-level; survives across requests)
# ═══════════════════════════════════════════════════════════════════════════════
_batch_state: dict = {}


@bp_perf.route('/batch/run', methods=['POST'])
def run_batch_perf_legacy():
    """Legacy redirect: now use /batch/launch."""
    return jsonify({'error': 'Use POST /api/v1/perf/batch/launch instead. See /batch/status/<id> for progress.'}), 410


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
        from .auth import get_current_user_id
        current_uid = get_current_user_id()
        removed = _db.cleanup_perf_tasks(root, user_id=current_uid)
        if removed:
            logger.info(f'Cleaned up {removed} stale perf task(s) from DB')
        items, total, available_models, available_datasets = _db.query_perf_tasks(
            search=search,
            filter_model=filter_model,
            filter_dataset=filter_dataset,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
            user_id=current_uid,
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
    from .auth import get_current_user_id
    from ..utils.process import get_user_slots
    uid = get_current_user_id()
    if not try_reserve_slot(task_id, 'perf', model=model, user_id=uid):
        max_perf = int(os.environ.get('MAX_PERF_PER_USER', '2'))
        slots = get_user_slots(uid)
        running = slots['perf']['used']
        return jsonify({
            'error': f'你的压测任务已达上限（{running}/{max_perf}），请等待完成后再试',
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
        # Default read timeout: prevent hanging on stalled streams
        if perf_args.read_timeout is None:
            perf_args.read_timeout = 300

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
            _mark_perf_completed(task_id)

            # Write to SQLite
            try:
                from datetime import datetime

                from .. import db as _db
                from .auth import get_current_user_id
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
                    user_id=get_current_user_id(),
                )
            except Exception as e:
                logger.warning(f'Failed to write perf to SQLite (non-fatal): {e}')

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


@bp_perf.route('/launch', methods=['POST'])
def launch_performance_test():
    """Non-blocking performance benchmark launch — returns immediately.

    Same request body as /invoke, but runs the benchmark in a background
    thread and returns the task_id immediately.  The client should poll
    /progress and use the SSE log stream (/log/stream) for real-time updates.
    """
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

    model = data.get('model', '')
    from .auth import get_current_user_id
    from ..utils.process import get_user_slots
    uid = get_current_user_id()
    if not try_reserve_slot(task_id, 'perf', model=model, user_id=uid):
        max_perf = int(os.environ.get('MAX_PERF_PER_USER', '2'))
        slots = get_user_slots(uid)
        running = slots['perf']['used']
        return jsonify({
            'error': f'你的压测任务已达上限（{running}/{max_perf}），请等待完成后再试',
            'running': running,
            'max': max_perf,
        }), 429

    try:
        api_type = data.get('api', 'openai')
        required_fields = ['model']
        if not api_type.startswith('local'):
            required_fields.append('url')
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400

        if 'api' not in data:
            data['api'] = 'openai'

        perf_args = PerfArguments.from_dict(data)
        perf_args.no_timestamp = True
        perf_args.outputs_dir = os.path.join(OUTPUT_DIR, task_id)
        perf_args.name = 'perf'
        perf_args.enable_progress_tracker = True
        perf_args.no_test_connection = True
        if perf_args.read_timeout is None:
            perf_args.read_timeout = 300

        os.makedirs(perf_args.outputs_dir, exist_ok=True)
        try:
            save_data = {k: v for k, v in data.items() if k != 'api_key'}
            config_file = os.path.join(perf_args.outputs_dir, 'task_config.json')
            with open(config_file, 'w') as f:
                json.dump(save_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f'[{task_id}] Failed to save task config: {e}')

        create_log_file(task_id, os.path.join('perf', 'benchmark.log'))

        logger.info(f'[{task_id}] Launching performance benchmark for model: {perf_args.model}')
        logger.info(f'[{task_id}] URL: {perf_args.url}')

        # ── Launch in background thread ──────────────────────────────
        import threading
        app = current_app._get_current_object()

        def _run():
            try:
                with app.app_context():
                    result = run_in_subprocess(
                        run_perf_wrapper, perf_args,
                        task_id=task_id, task_type='perf', model=perf_args.model
                    )
                    table_str = _build_perf_table(result, api_type=perf_args.api)
                    logger.info(f'[{task_id}] Task completed successfully')
                    _mark_perf_completed(task_id)

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
                            task_id=task_id, model=perf_args.model, api=perf_args.api,
                            dataset=perf_args.dataset_label or perf_args.dataset or 'N/A',
                            runs=runs, has_report=has_report,
                            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            user_id=uid,
                        )
                    except Exception as e:
                        logger.error(f'[{task_id}] Failed to write perf to SQLite (data remains on disk, will backfill on restart): {e}')
            except Exception as e:
                logger.error(f'[{task_id}] Background perf failed: {e}', exc_info=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return jsonify({'task_id': task_id, 'status': 'launched'}), 202

    except Exception as e:
        unregister_process(task_id)
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] [{task_id}] Launch setup failed: {e}', exc_info=True)
        return jsonify({
            'status': 'error', 'task_id': task_id,
            'error': 'Failed to start performance test', 'error_id': error_id,
        }), 500


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
    from .auth import get_current_user_id
    from ..utils.process import get_user_slots
    uid = get_current_user_id()
    if not try_reserve_slot(task_id, 'perf', model=model, user_id=uid):
        max_perf = int(os.environ.get('MAX_PERF_PER_USER', '2'))
        slots = get_user_slots(uid)
        running = slots['perf']['used']
        return jsonify({
            'error': f'你的压测任务已达上限（{running}/{max_perf}），请等待完成后再试',
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
            _mark_perf_completed(task_id)

            # Update SQLite
            try:
                from datetime import datetime

                from .. import db as _db
                from .auth import get_current_user_id
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
                    user_id=get_current_user_id(),
                )
            except Exception as e:
                logger.warning(f'Failed to write perf to SQLite (non-fatal): {e}')

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

    # Verify ownership
    from .auth import get_current_user_id
    from .. import db as _db
    owner = _db._get_conn().execute(
        'SELECT user_id FROM perf_tasks WHERE task_id = ?', (task_id,)
    ).fetchone()
    if owner and owner[0] != get_current_user_id():
        return jsonify({'error': 'Task not found'}), 404

    try:
        shutil.rmtree(task_dir)
        logger.info(f'Deleted perf task: {task_id}')

        # Sync SQLite
        try:
            _db.delete_perf_task(task_id, user_id=get_current_user_id())
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
    # Support resume: client can pass last_pos (byte offset) to skip already-seen content
    try:
        initial_pos = int(request.args.get('last_pos', 0))
    except (ValueError, TypeError):
        initial_pos = 0

    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    log_file = os.path.join(OUTPUT_DIR, task_id, 'perf', 'benchmark.log')

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


@bp_perf.route('/compare', methods=['GET'])
def compare_perf_reports():
    """Get combined perf metrics for side-by-side comparison.

    Query params:
        task_ids (str): comma-separated task IDs
    """
    task_ids_str = request.args.get('task_ids', '')
    if not task_ids_str:
        return jsonify({'error': 'task_ids is required'}), 400

    task_ids = [t.strip() for t in task_ids_str.split(',') if t.strip()]
    if not task_ids:
        return jsonify({'error': 'task_ids is required'}), 400

    from datetime import datetime

    tasks_data = []
    for tid in task_ids:
        try:
            validate_task_id(tid)
        except ValueError:
            continue

        task_dir = os.path.join(OUTPUT_DIR, tid)
        perf_dir = os.path.join(task_dir, 'perf')
        if not os.path.isdir(perf_dir):
            tasks_data.append({
                'task_id': tid,
                'model': os.path.basename(task_dir) if os.path.isdir(task_dir) else tid,
                'dataset': 'N/A',
                'api': 'N/A',
                'runs': [],
                'error': '无压测数据 (perf 目录不存在)',
            })
            continue

        model_name = 'N/A'
        dataset_name = 'N/A'
        api_type = 'N/A'
        for sub in sorted(os.listdir(perf_dir)):
            sub_dir = os.path.join(perf_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            args_file = os.path.join(sub_dir, 'benchmark_args.json')
            if os.path.isfile(args_file):
                try:
                    with open(args_file) as f:
                        args = json.load(f)
                    model_name = args.get('model', 'N/A')
                    dataset_name = args.get('dataset_label') or args.get('dataset', 'N/A')
                    api_type = args.get('api', 'N/A')
                except Exception:
                    pass
                break

        runs = []
        for sub in sorted(os.listdir(perf_dir)):
            sub_dir = os.path.join(perf_dir, sub)
            if not os.path.isdir(sub_dir):
                continue

            summary_file = os.path.join(sub_dir, 'benchmark_summary.json')
            if not os.path.isfile(summary_file):
                continue

            try:
                with open(summary_file) as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

            percentiles = []
            percentile_file = os.path.join(sub_dir, 'benchmark_percentile.json')
            if os.path.isfile(percentile_file):
                try:
                    with open(percentile_file) as f:
                        percentiles = json.load(f)
                except Exception:
                    pass

            throughput = {}
            throughput_file = os.path.join(sub_dir, 'workload_throughput.json')
            if os.path.isfile(throughput_file):
                try:
                    with open(throughput_file) as f:
                        throughput = json.load(f)
                except Exception:
                    pass

            runs.append({
                'run_name': sub,
                'summary': summary,
                'percentiles': percentiles,
                'throughput': throughput,
            })

        if not runs:
            # Include task even without benchmark data, so user sees all selected
            tasks_data.append({
                'task_id': tid,
                'model': model_name,
                'dataset': dataset_name,
                'api': api_type,
                'runs': [],
                'error': '无压测数据 (benchmark_summary.json 不存在)',
            })
            continue

        tasks_data.append({
            'task_id': tid,
            'model': model_name,
            'dataset': dataset_name,
            'api': api_type,
            'runs': runs,
        })

    if not tasks_data:
        return jsonify({'error': 'No valid perf reports found'}), 404

    return jsonify({
        'meta': {
            'generated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'task_count': len(tasks_data),
        },
        'tasks': tasks_data,
    }), 200


# --------------------------------------------------------------------------- #
# Compare report save / list / delete                                          #
# --------------------------------------------------------------------------- #

@bp_perf.route('/compare/save', methods=['POST'])
def save_compare_report():
    """Save a compare report (task IDs snapshot).

    JSON body: {"name": "...", "task_ids": ["..."], "backend": "LLM"|"Perf", "root_path": "..."}
    """
    data = request.get_json()
    if not data or not data.get('task_ids'):
        return jsonify({'error': 'task_ids is required'}), 400

    name = data.get('name', '').strip()
    task_ids = data['task_ids']
    backend_type = data.get('backend', 'Perf')
    root_path = data.get('root_path', '')
    if not isinstance(task_ids, list) or len(task_ids) < 2:
        return jsonify({'error': 'task_ids must be a list of at least 2'}), 400

    if not name:
        name = f'对比报告 ({len(task_ids)} 个模型)'

    try:
        from .. import db as _db
        from .auth import get_current_user_id
        report_id = _db.save_compare_report(name, json.dumps(task_ids), len(task_ids), backend_type, root_path, user_id=get_current_user_id())
        return jsonify({'id': report_id, 'name': name, 'task_count': len(task_ids)}), 201
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to save compare report: {e}', exc_info=True)
        return jsonify({'error': 'Failed to save', 'error_id': error_id}), 500


@bp_perf.route('/compare/saved', methods=['GET'])
def list_compare_reports():
    """List all saved compare reports."""
    try:
        from .. import db as _db
        from .auth import get_current_user_id
        reports = _db.list_compare_reports(user_id=get_current_user_id())
        return jsonify({'reports': reports}), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to list compare reports: {e}', exc_info=True)
        return jsonify({'error': 'Failed to list', 'error_id': error_id}), 500


@bp_perf.route('/compare/saved/<int:report_id>', methods=['DELETE'])
def delete_compare_report(report_id: int):
    """Delete a saved compare report."""
    try:
        from .. import db as _db
        from .auth import get_current_user_id
        deleted = _db.delete_compare_report(report_id, user_id=get_current_user_id())
        if deleted:
            return jsonify({'ok': True}), 200
        return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to delete compare report: {e}', exc_info=True)
        return jsonify({'error': 'Failed to delete', 'error_id': error_id}), 500


def _download_llm_compare(report: dict, task_ids: list, timestamp: str):
    """Generate LLM comparison HTML download."""
    from .. import db as _db

    # Fetch scores from eval_reports DB
    conn = _db._get_conn()
    rows = []
    for name in task_ids:
        row = conn.execute(
            'SELECT model_name, dataset_name, score, num_samples, timestamp FROM eval_reports WHERE task_id = ?',
            (name,)
        ).fetchone()
        if row:
            rows.append({
                'model': row[0],
                'dataset': row[1],
                'score': row[2],
                'samples': row[3],
                'ts': row[4] or '',
            })
        else:
            rows.append({
                'model': name, 'dataset': '-', 'score': None, 'samples': 0, 'ts': ''
            })

    rows_html = ''
    for r in rows:
        score_val = r.get('score')
        s = f'{score_val:.4f}' if score_val is not None else '-'
        rows_html += f'<tr><td>{r["model"]}</td><td>{r["dataset"]}</td><td>{r["samples"]}</td><td>{s}</td><td>{r["ts"]}</td></tr>\n'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模型评估对比报告 — {report['name']}</title>
<style>
:root {{ --bg:#f8f9fc; --card:#fff; --ink:#1a1a2e; --muted:#5a5a7a; --line:#e3e8ef; --accent:#5B3FD6; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--ink); line-height:1.6; font-size:14px; }}
.wrap {{ max-width:900px; margin:0 auto; padding:28px 20px 60px; }}
header {{ background:linear-gradient(135deg,#1e3a8a,#2563eb); color:#fff; border-radius:12px; padding:20px 24px; margin-bottom:20px; }}
header h1 {{ font-size:20px; margin-bottom:4px; }}
header .sub {{ font-size:12px; opacity:.85; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:20px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:10px; padding:12px 14px; }}
.kpi .v {{ font-size:20px; font-weight:700; }}
.kpi .l {{ font-size:11px; color:var(--muted); }}
.section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin-bottom:16px; }}
.section h2 {{ font-size:15px; margin-bottom:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; border-bottom:2px solid var(--line); font-size:12px; }}
tr:hover {{ background:#f0f4ff; }}
footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:30px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>模型评估对比报告</h1>
  <div class="sub">{report['name']} · {len(rows)} 个模型 · {timestamp}</div>
</header>
<div class="kpis">
  <div class="kpi"><div class="v">{len(rows)}</div><div class="l">评估模型数</div></div>
</div>
<div class="section">
  <h2>评估得分对比</h2>
  <table>
    <thead><tr><th>模型</th><th>数据集</th><th>样本数</th><th>得分</th><th>创建时间</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
<footer>由 EvalPerf 生成 · {timestamp}</footer>
</div>
</body>
</html>'''

    from flask import Response
    report_id = report['id']
    return Response(html, mimetype='text/html', headers={'Content-Disposition': f'attachment; filename=llm_compare_{report_id}.html'})


@bp_perf.route('/compare/saved/<int:report_id>/download', methods=['GET'])
def download_compare_report(report_id: int):
    """Download a saved compare report as a self-contained HTML file."""
    try:
        from .. import db as _db
        from .auth import get_current_user_id
        reports = _db.list_compare_reports(user_id=get_current_user_id())
        report = next((r for r in reports if r['id'] == report_id), None)
        if not report:
            return jsonify({'error': 'Report not found'}), 404

        task_ids = json.loads(report['task_ids'])
        timestamp = report['created_at']
        backend_type = report.get('backend', 'Perf')

        if backend_type == 'LLM':
            return _download_llm_compare(report, task_ids, timestamp)

        # Collect per-task summary data (Perf)
        tasks_data = []
        for tid in task_ids:
            task_dir = os.path.join(OUTPUT_DIR, tid)
            perf_dir = os.path.join(task_dir, 'perf')
            if not os.path.isdir(perf_dir):
                continue
            model_name = tid
            found_summary = None
            for sub in sorted(os.listdir(perf_dir)):
                sub_dir = os.path.join(perf_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue
                args_file = os.path.join(sub_dir, 'benchmark_args.json')
                if os.path.isfile(args_file):
                    try:
                        with open(args_file) as f:
                            args = json.load(f)
                        model_name = args.get('model', tid)
                    except Exception:
                        pass
                summary_file = os.path.join(sub_dir, 'benchmark_summary.json')
                if os.path.isfile(summary_file):
                    try:
                        with open(summary_file) as f:
                            found_summary = json.load(f)
                    except Exception:
                        pass
                if found_summary:
                    tasks_data.append({'model': model_name, 'summary': found_summary})
                    break

        if not tasks_data:
            return jsonify({'error': 'No valid task data found'}), 404

        # Generate HTML
        rows_html = ''
        for t in tasks_data:
            s = t['summary']
            sr = (s.get('Success Requests', 0) / s.get('Total Requests', 1) * 100) if s.get('Total Requests') else 100
            rows_html += f'''<tr>
    <td>{t['model']}</td>
    <td>{s.get('Concurrency', '-')}</td>
    <td>{s.get('Total Requests', '-')}</td>
    <td>{s.get('Success Requests', '-')}</td>
    <td>{s.get('Failed Requests', 0)}</td>
    <td>{sr:.1f}%</td>
    <td>{(s.get('Req Throughput (req/s)', 0) * 60):.1f}</td>
    <td>{(s.get('Output Throughput (tok/s)', 0) * 60):.0f}</td>
    <td>{s.get('Avg Latency (s)', 0):.2f}s</td>
    <td>{s.get('TTFT (ms)', 0):.0f}ms</td>
    <td>{s.get('TPOT (ms)', 0):.1f}ms</td>
    <td>{s.get('Output Throughput (tok/s)', 0):.1f}</td>
    <td>{s.get('Total Throughput (tok/s)', 0):.1f}</td>
</tr>'''

        # Generate chart data as JSON
        models_json = []
        total_reqs = 0
        total_succ = 0
        total_latency = 0
        total_output_tps = 0
        for t in tasks_data:
            s = t['summary']
            sr = (s.get('Success Requests', 0) / s.get('Total Requests', 1) * 100) if s.get('Total Requests') else 100
            total_reqs += s.get('Total Requests', 0)
            total_succ += s.get('Success Requests', 0)
            total_latency += s.get('Avg Latency (s)', 0)
            total_output_tps += s.get('Output Throughput (tok/s)', 0)
            models_json.append({
                'name': t['model'],
                'rpm': round(s.get('Req Throughput (req/s)', 0) * 60, 1),
                'tpm': round(s.get('Output Throughput (tok/s)', 0) * 60, 0),
                'latency': round(s.get('Avg Latency (s)', 0), 2),
                'ttft': round(s.get('TTFT (ms)', 0), 0),
                'tpot': round(s.get('TPOT (ms)', 0), 1),
                'success_rate': round(sr, 1),
            })

        avg_sr = (total_succ / total_reqs * 100) if total_reqs else 100
        avg_lat = total_latency / len(tasks_data) if tasks_data else 0
        avg_tps = total_output_tps / len(tasks_data) if tasks_data else 0

        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>压测对比报告 — {report['name']}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg:#f8f9fc; --card:#fff; --ink:#1a1a2e; --muted:#5a5a7a; --line:#e3e8ef; --accent:#5B3FD6; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--ink); line-height:1.6; font-size:14px; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:28px 20px 60px; }}
header {{ background:linear-gradient(135deg,#1e3a8a,#2563eb); color:#fff; border-radius:12px; padding:20px 24px; margin-bottom:20px; }}
header h1 {{ font-size:20px; margin-bottom:4px; }}
header .sub {{ font-size:12px; opacity:.85; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:20px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:10px; padding:12px 14px; }}
.kpi .v {{ font-size:20px; font-weight:700; }}
.kpi .l {{ font-size:11px; color:var(--muted); }}
.section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin-bottom:16px; }}
.section h2 {{ font-size:15px; margin-bottom:12px; color:var(--ink); }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; margin-bottom:16px; }}
.chart-box {{ position:relative; height:260px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--muted); font-weight:600; border-bottom:2px solid var(--line); font-size:12px; }}
tr:hover {{ background:#f0f4ff; }}
footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:30px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>压测对比报告</h1>
  <div class="sub">{report['name']} · {len(tasks_data)} 个模型 · 生成时间 {timestamp}</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">{len(tasks_data)}</div><div class="l">已选任务</div></div>
  <div class="kpi"><div class="v">{total_reqs}</div><div class="l">总请求数</div></div>
  <div class="kpi"><div class="v">{avg_sr:.1f}%</div><div class="l">平均成功率</div></div>
  <div class="kpi"><div class="v">{avg_lat:.2f}s</div><div class="l">平均延迟</div></div>
  <div class="kpi"><div class="v">{avg_tps:.1f}</div><div class="l">平均输出 TPS</div></div>
</div>

<div class="section">
  <h2>指标对比表</h2>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>模型</th><th>并发</th><th>请求数</th><th>成功</th><th>失败</th><th>成功率</th>
      <th>RPM</th><th>TPM</th><th>Avg延迟</th><th>TTFT</th><th>TPOT</th><th>输出tok/s</th><th>总tok/s</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>

<div class="charts">
  <div class="section" style="margin-bottom:0"><h2>吞吐能力 (RPM)</h2><div class="chart-box"><canvas id="cRPM"></canvas></div></div>
  <div class="section" style="margin-bottom:0"><h2>吞吐能力 (TPM)</h2><div class="chart-box"><canvas id="cTPM"></canvas></div></div>
  <div class="section" style="margin-bottom:0"><h2>Avg 延迟 (s)</h2><div class="chart-box"><canvas id="cLatency"></canvas></div></div>
  <div class="section" style="margin-bottom:0"><h2>TTFT 首字延迟 (ms)</h2><div class="chart-box"><canvas id="cTTFT"></canvas></div></div>
  <div class="section" style="margin-bottom:0"><h2>TPOT 生成间隔 (ms)</h2><div class="chart-box"><canvas id="cTPOT"></canvas></div></div>
  <div class="section" style="margin-bottom:0"><h2>成功率 (%)</h2><div class="chart-box"><canvas id="cSuccess"></canvas></div></div>
</div>

<footer>由 EvalPerf 生成 · {timestamp}</footer>
</div>

<script>
var MODELS = {json.dumps(models_json, ensure_ascii=False)};
var labels = MODELS.map(function(m){{ return m.name; }});
var palette = ['#5B3FD6','#0F9C7E','#f59e0b','#ef4444','#3b82f6','#8b5cf6','#ec4899','#14b8a6'];

function mkBar(id, key, fmt, unit) {{
  if(typeof Chart === 'undefined') return;
  new Chart(document.getElementById(id), {{
    type:'bar',
    data:{{ labels:labels, datasets:[{{ label:key, data:MODELS.map(function(m){{ return m[key]; }}), backgroundColor:MODELS.map(function(_,i){{ return palette[i%palette.length]; }}) }}] }},
    options:{{ responsive:true, maintainAspectRatio:false,
      plugins:{{ tooltip:{{ callbacks:{{ label:function(c){{ return fmt ? fmt(c.raw) : c.raw+(unit||''); }} }} }} }},
      scales:{{ y:{{ title:{{ display:!!unit, text:unit||'' }} }} }}
    }}
  }});
}}

mkBar('cRPM', 'rpm', null, 'req/min');
mkBar('cTPM', 'tpm', null, 'tok/min');
mkBar('cLatency', 'latency', null, 's');
mkBar('cTTFT', 'ttft', null, 'ms');
mkBar('cTPOT', 'tpot', null, 'ms');
mkBar('cSuccess', 'success_rate', function(v){{ return v+'%'; }}, '%');
</script>
</body>
</html>'''

        from flask import Response
        filename = f'perf_compare_{report_id}.html'
        return Response(
            html,
            mimetype='text/html',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to generate compare report: {e}', exc_info=True)
        return jsonify({'error': 'Failed to generate report', 'error_id': error_id}), 500
