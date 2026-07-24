"""Audio evaluation API blueprint."""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, send_file

logger = logging.getLogger(__name__)

bp_audio = Blueprint('audio', __name__, url_prefix='/api/v1/audio')

OUTPUT_DIR = Path(os.getenv('EVALSCOPE_OUTPUT_DIR', './outputs'))


@bp_audio.route('/reports', methods=['GET'])
def list_audio_reports():
    """List all Audio evaluation reports."""
    reports = []

    if not OUTPUT_DIR.exists():
        return jsonify({'reports': reports})

    for task_dir in OUTPUT_DIR.iterdir():
        if not task_dir.is_dir():
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
            created_at = datetime.fromtimestamp(
                float(results.get('timestamp') or task_dir.stat().st_mtime)
            ).isoformat()
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

        reports.append(report)

    reports.sort(key=lambda r: r.get('created_at', ''), reverse=True)
    return jsonify({'reports': reports})


@bp_audio.route('/report/<task_id>', methods=['GET'])
def get_audio_report(task_id: str):
    """Get full audio evaluation report."""
    task_dir = OUTPUT_DIR / task_id
    results_file = task_dir / 'results.json'

    if not results_file.exists():
        return jsonify({'error': 'Report not found'}), 404

    try:
        with open(results_file, encoding='utf-8') as f:
            results = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return jsonify({'error': f'Failed to read report: {e}'}), 500

    # Build media URLs for audio files
    for sample in results.get('per_sample', []):
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
    """Serve file from audio output directory with path traversal protection."""
    task_dir = os.path.realpath(os.path.join(OUTPUT_DIR, task_id))
    safe_path = os.path.realpath(os.path.join(task_dir, filename))

    if not safe_path.startswith(task_dir + os.sep) and safe_path != task_dir:
        return jsonify({'error': 'Invalid path'}), 403

    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(safe_path)
