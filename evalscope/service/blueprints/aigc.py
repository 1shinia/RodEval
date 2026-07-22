"""AIGC evaluation API blueprint."""
import json
import logging
import os
import time
from datetime import datetime
from flask import Blueprint, jsonify, request, send_file
from pathlib import Path
from typing import Any, Dict

from evalscope.backend.aigc_eval.backend_manager import AIGCBackendManager
from evalscope.service.utils.process import register_process, try_reserve_slot, unregister_process

logger = logging.getLogger(__name__)

bp_aigc = Blueprint('aigc', __name__, url_prefix='/api/v1/aigc')

OUTPUT_DIR = Path(os.getenv('EVALSCOPE_OUTPUT_DIR', './outputs'))

# Max concurrent AIGC tasks (read from env, default 1)
MAX_CONCURRENT_AIGC = int(os.environ.get('MAX_CONCURRENT_AIGC', '1'))


@bp_aigc.route('/invoke', methods=['POST'])
def run_aigc_evaluation():
    """Submit AIGC evaluation task."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        task_id = request.headers.get('EvalScope-Task-Id')
        if not task_id:
            return jsonify({'error': 'Missing EvalScope-Task-Id header'}), 400

        model = data.get('model', {}).get('model_name_or_path', 'unknown')

        # Reserve slot for concurrent control
        if not try_reserve_slot(task_id, 'aigc', model):
            return jsonify({'error': f'Max concurrent AIGC tasks reached ({MAX_CONCURRENT_AIGC})'}), 429

        try:
            # Build configuration
            config = _build_aigc_config(data, task_id)

            # Execute evaluation
            result = _execute_aigc_task(task_id, config)

            return jsonify({
                'task_id': task_id,
                'status': 'completed',
                'result': result,
            })

        finally:
            unregister_process(task_id)

    except Exception as e:
        logger.exception('AIGC evaluation failed')
        return jsonify({'error': str(e)}), 500


@bp_aigc.route('/progress', methods=['GET'])
def get_aigc_progress():
    """Get AIGC evaluation progress."""
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'Missing task_id parameter'}), 400

    progress_file = OUTPUT_DIR / task_id / 'progress.json'

    if not progress_file.exists():
        # Check if task directory exists
        task_dir = OUTPUT_DIR / task_id
        if task_dir.exists():
            return jsonify({'percent': 0.0, 'status': 'running'})
        else:
            return jsonify({'error': 'Task not found'}), 404

    with open(progress_file) as f:
        progress = json.load(f)

    return jsonify(progress)


@bp_aigc.route('/stop', methods=['POST'])
def stop_aigc_evaluation():
    """Stop AIGC evaluation task."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    task_id = data.get('task_id')
    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400

    # Update progress file to mark as stopped
    progress_file = OUTPUT_DIR / task_id / 'progress.json'
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        progress['status'] = 'stopped'
        with open(progress_file, 'w') as f:
            json.dump(progress, f)

    unregister_process(task_id)
    return jsonify({'status': 'stopped'})


@bp_aigc.route('/media/<task_id>/<path:filename>', methods=['GET'])
def serve_media(task_id: str, filename: str):
    """Serve generated media files."""
    safe_path = os.path.realpath(os.path.join(OUTPUT_DIR, task_id, 'media', filename))

    # Path traversal protection
    if not safe_path.startswith(os.path.realpath(OUTPUT_DIR)):
        return jsonify({'error': 'Invalid path'}), 403

    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(safe_path)


@bp_aigc.route('/thumbnails/<task_id>/<path:filename>', methods=['GET'])
def serve_thumbnail(task_id: str, filename: str):
    """Serve thumbnail files."""
    safe_path = os.path.realpath(os.path.join(OUTPUT_DIR, task_id, 'thumbnails', filename))

    # Path traversal protection
    if not safe_path.startswith(os.path.realpath(OUTPUT_DIR)):
        return jsonify({'error': 'Invalid path'}), 403

    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(safe_path)


@bp_aigc.route('/report', methods=['GET'])
def get_aigc_report():
    """Load AIGC evaluation report results."""
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify({'error': 'Missing task_id parameter'}), 400

    task_dir = OUTPUT_DIR / task_id
    if not task_dir.exists():
        return jsonify({'error': 'Task not found'}), 404

    results_file = task_dir / 'results.json'
    if not results_file.exists():
        return jsonify({'error': 'Results file not found'}), 404

    with open(results_file) as f:
        results = json.load(f)

    # Build media URLs for each sample
    for sample in results.get('per_sample', []):
        img_path = sample.get('image_path', '')
        thumb_path = sample.get('thumbnail_path', '')
        video_path = sample.get('video_path', '')
        if img_path:
            sample['url'] = f'/api/v1/aigc/file/{task_id}/{img_path}'
        if video_path:
            sample['video_url'] = f'/api/v1/aigc/file/{task_id}/{video_path}'
        sample['thumbnail_url'] = f'/api/v1/aigc/file/{task_id}/{thumb_path}' if thumb_path else ''

    return jsonify(results)


@bp_aigc.route('/file/<task_id>/<path:filename>', methods=['GET'])
def serve_file(task_id: str, filename: str):
    """Serve any file from the task output directory with path traversal protection."""
    task_dir = os.path.realpath(os.path.join(OUTPUT_DIR, task_id))
    safe_path = os.path.realpath(os.path.join(task_dir, filename))

    # Path traversal protection
    if not safe_path.startswith(task_dir + os.sep) and safe_path != task_dir:
        return jsonify({'error': 'Invalid path'}), 403

    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(safe_path)


@bp_aigc.route('/benchmarks', methods=['GET'])
def get_aigc_benchmarks():
    """Get list of AIGC benchmarks."""
    benchmarks = [
        {
            'name': 'DrawBench',
            'description': '200 prompts across 11 categories for text-to-image evaluation',
            'prompts': 200,
        },
        {
            'name': 'COCO Captions',
            'description': 'COCO dataset captions for image generation',
            'prompts': 5000,
        },
        {
            'name': 'PartiPrompts',
            'description': '1600+ diverse prompts for challenging text-to-image generation',
            'prompts': 1600,
        },
    ]
    return jsonify(benchmarks)


@bp_aigc.route('/reports', methods=['GET'])
def list_aigc_reports():
    """List all AIGC evaluation reports."""
    reports = []

    if not OUTPUT_DIR.exists():
        return jsonify({'reports': reports})

    # Scan task directories
    for task_dir in OUTPUT_DIR.iterdir():
        if not task_dir.is_dir():
            continue

        results_file = task_dir / 'results.json'
        if not results_file.exists():
            continue

        try:
            with open(results_file) as f:
                results = json.load(f)

            # Extract summary info
            report = {
                'task_id': task_dir.name,
                'model_name': results.get('model', 'unknown'),
                'model_type': results.get('model_type', 'txt2img'),
                'total_images': results.get('num_samples', 0),
                'clip_score_mean': results.get('metrics', {}).get('clip_score_mean'),
                'fid': results.get('metrics', {}).get('fid'),
                'inception_score': results.get('metrics', {}).get('inception_score'),
                'created_at': datetime.fromtimestamp(float(results.get('timestamp')
                                                           or task_dir.stat().st_mtime)).isoformat(),
            }
            reports.append(report)
        except Exception as e:
            logger.warning(f'Failed to load report from {task_dir}: {e}')

    # Sort by created_at descending
    reports.sort(key=lambda x: x['created_at'], reverse=True)

    return jsonify({'reports': reports})


def _build_aigc_config(data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Build AIGC configuration from request data."""
    output_dir = OUTPUT_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'tool': data.get('tool', 'txt2img'),
        'model': {
            'model_name_or_path': data.get('model', {}).get('model_name_or_path', ''),
            'model_type': data.get('model', {}).get('model_type', 'txt2img'),
            'api_base': data.get('model', {}).get('api_base'),
            'api_key': data.get('model', {}).get('api_key'),
            'device': data.get('model', {}).get('device', 'cuda'),
            'dtype': data.get('model', {}).get('dtype', 'float16'),
        },
        'generate': {
            'width': data.get('generate', {}).get('width', 512),
            'height': data.get('generate', {}).get('height', 512),
            'num_inference_steps': data.get('generate', {}).get('num_inference_steps', 50),
            'guidance_scale': data.get('generate', {}).get('guidance_scale', 7.5),
            'negative_prompt': data.get('generate', {}).get('negative_prompt', ''),
            'seed': data.get('generate', {}).get('seed', 42),
            'batch_size': data.get('generate', {}).get('batch_size', 1),
        },
        'eval': {
            'metrics': data.get('eval', {}).get('metrics', ['clip_score']),
            'prompt_dataset': data.get('eval', {}).get('prompt_dataset', 'drawbench'),
            'prompt_limit': data.get('eval', {}).get('prompt_limit', 100),
            'reference_dir': data.get('eval', {}).get('reference_dir'),
            'reference_video_dir': data.get('eval', {}).get('reference_video_dir'),
            'custom_dataset_path': data.get('eval', {}).get('custom_dataset_path'),
            'output_dir': str(output_dir),
        },
    }

    return config


def _execute_aigc_task(task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute AIGC evaluation task."""
    # Note: register_process expects (task_id, proc, task_type, model)
    # but we're running synchronously here, so we just use try_reserve_slot
    # which already registered a placeholder. No need to call register_process again.

    manager = AIGCBackendManager(config)
    result = manager.run()

    return result
