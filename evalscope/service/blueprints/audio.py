"""Audio evaluation API blueprint."""
import json
import logging
import os
from pathlib import Path

from flask import Blueprint, jsonify, send_file
from evalscope.service.utils.log import (
    OUTPUT_DIR as _OUTPUT_DIR,
    resolve_task_dir,
    resolve_task_file,
    validate_task_id,
)
from evalscope.service.time_utils import epoch_to_utc_iso

logger = logging.getLogger(__name__.replace('evalscope', 'evalperf'))

bp_audio = Blueprint('audio', __name__, url_prefix='/api/v1/audio')

OUTPUT_DIR = Path(_OUTPUT_DIR)


def _require_task_access(task_id: str):
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    from .auth import check_task_artifact_access
    if not check_task_artifact_access(task_id, ('eval_reports', 'task_state')):
        return jsonify({'error': 'Report not found'}), 404
    return None


@bp_audio.route('/reports', methods=['GET'])
def list_audio_reports():
    """List all Audio evaluation reports (filtered by current user)."""
    reports = []

    if not OUTPUT_DIR.exists():
        return jsonify({'reports': reports})

    # Get current user's Audio task_ids from eval_reports table
    from .auth import get_current_user_id
    from ..db import _get_conn
    current_uid = get_current_user_id()
    conn = _get_conn()
    user_task_ids = {
        r[0] for r in conn.execute(
            "SELECT task_id FROM eval_reports WHERE user_id = ? AND eval_backend = 'AudioEval'",
            (current_uid,)
        ).fetchall()
    }

    # Scan only user's task directories
    for task_dir in OUTPUT_DIR.iterdir():
        if not task_dir.is_dir():
            continue
        if task_dir.name not in user_task_ids:
            continue

        results_file = task_dir / 'results.json'
        if not results_file.exists():
            continue

        try:
            with open(results_file, encoding='utf-8') as f:
                results = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        tool = results.get('tool', 'unknown')
        if tool not in ('asr', 'tts'):
            continue

        metrics = results.get('metrics', {})

        created_at = ''
        try:
            created_at = epoch_to_utc_iso(float(results.get('timestamp') or task_dir.stat().st_mtime))
        except (OSError, ValueError):
            pass

        report = {
            'task_id': task_dir.name,
            'tool': tool,
            'model_name': results.get('model', 'unknown'),
            'created_at': created_at,
        }

        # ASR metrics
        if tool == 'asr':
            per_sample = results.get('per_sample', {})
            report['wer'] = metrics.get('wer')
            report['cer'] = metrics.get('cer')
            report['reference'] = per_sample.get('reference', '')[:50]
            report['hypothesis'] = per_sample.get('hypothesis', '')[:50]
            report['language'] = per_sample.get('language', '')

        # TTS info
        if tool == 'tts':
            report['num_samples'] = results.get('num_samples', 0)
            report['total_elapsed'] = results.get('total_elapsed_seconds', 0)
            # Closed-loop ASR evaluation metrics
            if metrics.get('wer_avg') is not None:
                report['wer'] = metrics['wer_avg']
            if metrics.get('cer_avg') is not None:
                report['cer'] = metrics['cer_avg']

        reports.append(report)

    reports.sort(key=lambda r: r.get('created_at', ''), reverse=True)
    return jsonify({'reports': reports})


@bp_audio.route('/report/<task_id>', methods=['GET'])
def get_audio_report(task_id: str):
    """Get full audio evaluation report."""
    denied = _require_task_access(task_id)
    if denied is not None:
        return denied
    task_dir = (OUTPUT_DIR / task_id).resolve()
    results_file = task_dir / 'results.json'

    if not results_file.exists():
        return jsonify({'error': 'Report not found'}), 404

    try:
        with open(results_file, encoding='utf-8') as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({'error': f'Failed to read report: {e}'}), 500

    # Build media URLs for audio files
    samples = results.get('per_sample', [])
    # Normalize: ASR stores single dict, TTS stores list of dicts
    if isinstance(samples, dict):
        samples = [samples]
        results['per_sample'] = samples
    for sample in samples:
        audio_path = sample.get('audio_path', '')
        if audio_path:
            if audio_path.startswith(str(task_dir)):
                rel_path = os.path.relpath(audio_path, task_dir)
            else:
                rel_path = audio_path
            sample['audio_url'] = f'/api/v1/audio/file/{task_id}/{rel_path}'

    return jsonify(results)


@bp_audio.route('/file/<task_id>/<path:filename>', methods=['GET'])
def serve_file(task_id: str, filename: str):
    """Serve generated audio artifacts from the task audio directory."""
    denied = _require_task_access(task_id)
    if denied is not None:
        return denied
    try:
        safe_path = resolve_task_file(task_id, filename, OUTPUT_DIR, allowed_subdirs=('audio',))
    except ValueError as e:
        if str(e) == 'File not found':
            return jsonify({'error': 'File not found'}), 404
        return jsonify({'error': 'Invalid path'}), 403
    return send_file(safe_path)


@bp_audio.route('/reports/<task_id>', methods=['DELETE'])
def delete_audio_report(task_id: str):
    """Delete an Audio evaluation report by task_id."""
    import shutil
    try:
        task_dir = resolve_task_dir(task_id, OUTPUT_DIR, must_exist=True)
    except ValueError as e:
        status = 404 if str(e) == 'Task not found' else 400
        return jsonify({'error': str(e)}), status

    # Destructive operations require durable ownership evidence; admin status
    # alone must not turn an arbitrary unindexed directory into a task.
    from .auth import get_current_user_id, check_task_artifact_access
    from .. import db as _db
    if not check_task_artifact_access(
        task_id, ('eval_reports', 'task_registry', 'task_state'), allow_admin_legacy=False
    ):
        return jsonify({'error': 'Report not found'}), 404

    shutil.rmtree(str(task_dir))
    logger.info(f'Deleted Audio report: {task_dir}')
    # Sync SQLite
    try:
        _db.delete_eval_report(task_id, user_id=get_current_user_id())
    except Exception as e:
        logger.warning(f'Failed to delete Audio task {task_id} from SQLite: {e}')
    return jsonify({'success': True})
