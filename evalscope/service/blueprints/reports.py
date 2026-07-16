"""Blueprint for report browsing and data access.

Exposes the file-system report data through a REST API so that the
React SPA frontend can load reports, predictions and analyses without
direct filesystem access.
"""
import json
import mimetypes
import os
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime
from flask import Blueprint, jsonify, request, send_file
from typing import List

from evalscope.constants import PLOTLY_CDN_URL, PLOTLY_THEME
from evalscope.report import ReportKey, get_data_frame
from evalscope.report.report import Report
from evalscope.report.visualization import (
    plot_multi_report_radar,
    plot_single_dataset_scores,
    plot_single_report_scores,
    plot_single_report_sunburst,
)
from evalscope.utils.data_utils import (
    get_acc_report_df,
    get_compare_report_df,
    get_model_prediction,
    get_report_analysis,
    load_multi_report,
    load_single_report,
    normalize_score,
    process_report_name,
    scan_for_report_folders,
)
from evalscope.utils.io_utils import OutputsStructure
from evalscope.utils.logger import get_logger
from ..utils import OUTPUT_DIR, validate_report_name, validate_root_path

logger = get_logger()

bp_reports = Blueprint('reports', __name__, url_prefix='/api/v1/reports')

_DEFAULT_ROOT = OUTPUT_DIR

# Allowed extensions for the media proxy (security: do not serve arbitrary files)
_MEDIA_EXTENSIONS = {
    # image
    '.jpg',
    '.jpeg',
    '.png',
    '.gif',
    '.webp',
    '.bmp',
    '.svg',
    '.ico',
    # video
    '.mp4',
    '.webm',
    '.ogg',
    '.ogv',
    '.mov',
    '.avi',
    '.mkv',
    # audio
    '.mp3',
    '.wav',
    '.flac',
    '.aac',
    '.m4a',
    '.opus',
}


@bp_reports.route('/media/file', methods=['GET'])
def serve_media_file():
    """Serve a local media file (image / audio / video) via HTTP.

    This proxy endpoint allows the browser to load server-side local file
    paths that are stored inside prediction records (e.g. video paths from
    MVBench datasets).

    Query params:
        path (str): Absolute path to the media file on the server.

    Security:
        - Only files with known media extensions are served.
        - The file must exist and be a regular file.
    """
    file_path = request.args.get('path', '').strip()
    if not file_path:
        return jsonify({'error': 'path parameter is required'}), 400

    # Normalise to absolute path and reject directory traversal
    file_path = os.path.realpath(file_path)

    # Security: restrict to OUTPUT_DIR to prevent arbitrary file reads
    allowed_dir = os.path.realpath(OUTPUT_DIR)
    if file_path != allowed_dir and not file_path.startswith(allowed_dir + os.sep):
        return jsonify({'error': 'Access denied: path must be within the outputs directory'}), 403

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _MEDIA_EXTENSIONS:
        return jsonify({'error': f'File type {ext!r} is not allowed'}), 403

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    return send_file(file_path, mimetype=mime_type)


def _root_path() -> str:
    # Priority: URL query param > app config (from --outputs CLI arg) > default
    from flask import current_app
    raw = request.args.get('root_path', current_app.config.get('OUTPUTS_ROOT') or _DEFAULT_ROOT)
    return validate_root_path(raw)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _df_to_records(df) -> list:
    """Convert a pandas DataFrame to a list of dicts, handling NaN."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient='records', force_ascii=False))


def _report_dir_exists(report_name: str, root: str) -> bool:
    """Check that the report's directory still exists on disk."""
    try:
        prefix, _, _ = process_report_name(report_name)
        return os.path.isdir(os.path.join(root, prefix))
    except Exception:
        return False


def _is_mteb_report(root: str, report_name: str) -> bool:
    """Check if a report is an MTEB/RAG evaluation (has results/ but not reports/)."""
    try:
        prefix, _, _ = process_report_name(report_name)
        task_dir = os.path.join(root, prefix)
        results_dir = os.path.join(task_dir, 'results')
        reports_dir = os.path.join(task_dir, 'reports')
        return os.path.isdir(results_dir) and not os.path.isdir(reports_dir)
    except Exception:
        return False


def _load_mteb_report_data(root: str, report_name: str, task_cfg: dict):
    """Build a simplified report list and datasets from MTEB results/ directory."""
    prefix, _, _ = process_report_name(report_name)
    task_dir = os.path.join(root, prefix)
    results_dir = os.path.join(task_dir, 'results')

    reports = []
    datasets = []
    if not os.path.isdir(results_dir):
        return reports, datasets

    # Get sample count from task config (same logic as _read_mteb_sample_count)
    limits = task_cfg.get('eval_config', {}).get('eval', {}).get('limits')

    for rroot, _, rfiles in os.walk(results_dir):
        for rf in rfiles:
            if not rf.endswith('.json') or rf == 'model_meta.json':
                continue
            fpath = os.path.join(rroot, rf)
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                task_name = data.get('task_name', rf.replace('.json', ''))
                main_score = None
                for split_data in data.get('scores', {}).values():
                    if isinstance(split_data, list) and split_data:
                        main_score = split_data[0].get('main_score')
                        break

                rel_path = os.path.relpath(fpath, results_dir)
                parts = rel_path.split(os.sep)
                model_name = parts[0].replace('eval__', '') if parts else 'unknown'

                if main_score is not None:
                    samples = limits if limits is not None else 0
                    datasets.append(task_name)
                    reports.append({
                        'name': task_name,
                        'dataset_name': task_name,
                        'model_name': model_name,
                        'score': main_score,
                        'analysis': '',
                        'metrics': [{
                            'name': 'main_score',
                            'score': main_score,
                            'num': samples,
                            'categories': [{
                                'name': 'default',
                                'score': main_score,
                                'num': samples,
                                'subsets': [{
                                    'name': 'test',
                                    'score': main_score,
                                    'num': samples,
                                }],
                            }],
                        }],
                    })
            except Exception:
                continue

    return reports, datasets


def _extract_timestamp(report_name: str, root: str) -> str:
    """Try to extract a timestamp from the report directory name or fall back to mtime."""
    try:
        prefix, _, _ = process_report_name(report_name)
        # Directory names typically look like "20260423_201338"
        for fmt in ('%Y%m%d_%H%M%S', '%Y%m%d'):
            try:
                dt = datetime.strptime(prefix, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        # Fall back to directory modification time
        dir_path = os.path.join(root, prefix)
        if os.path.isdir(dir_path):
            mtime = os.path.getmtime(dir_path)
            return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        pass
    return ''


def _build_report_meta(report_name: str, root: str) -> dict:
    """Load a report and return lightweight metadata for the list endpoint."""
    try:
        report_list, datasets, _ = load_single_report(root, report_name)
    except Exception:
        return None

    if not report_list:
        return None

    # Aggregate: use the first report's model_name; collect all dataset names
    first = report_list[0]
    total_num = 0
    dataset_names = []
    score_sum = 0.0
    for r in report_list:
        dataset_names.append(r.dataset_name)
        total_num += r.num or 0
        score_sum += r.score

    avg_score = round(score_sum / len(report_list), 4) if report_list else 0.0
    timestamp = _extract_timestamp(report_name, root)

    # Build per-dataset score mapping (normalized to 0-1 range)
    dataset_scores = {}
    for r in report_list:
        score = r.score
        if score is not None and score > 1:
            score = score / 100
        dataset_scores[r.dataset_name] = round(score, 4) if score is not None else None

    return {
        'name': report_name,
        'model_name': first.model_name,
        'dataset_name': ', '.join(dataset_names) if len(dataset_names) > 1 else
        (dataset_names[0] if dataset_names else ''),
        'score': avg_score,
        'dataset_scores': dataset_scores,
        'num_samples': total_num,
        'timestamp': timestamp,
        # keep individual scores for per-dataset filtering
        '_datasets': dataset_names,
        '_scores': [r.score for r in report_list],
    }


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@bp_reports.route('/list', methods=['GET'])
def list_reports():
    """Return a filterable, paginated list of reports with metadata.

    Uses SQLite for fast queries.  Falls back to filesystem scan if the
    DB is not initialised.

    Query params:
        root_path  (str):   output root directory (required)
        search     (str):   fuzzy search on model/dataset name
        models     (str):   semicolon-separated model filter
        datasets   (str):   semicolon-separated dataset filter
        score_min  (float): minimum score (0-1)
        score_max  (float): maximum score (0-1)
        sort_by    (str):   score / model / dataset / time (default: time)
        sort_order (str):   asc / desc (default: desc)
        page       (int):   page number (default: 1)
        page_size  (int):   items per page (default: 20)
    """
    try:
        root = _root_path()
        if not root or not os.path.isdir(root):
            return jsonify({'error': 'root_path is required and must be an existing directory'}), 400

        search = request.args.get('search', '').strip().lower()
        models_filter = request.args.get('models', '').strip()
        datasets_filter = request.args.get('datasets', '').strip()
        score_min = request.args.get('score_min', type=float)
        score_max = request.args.get('score_max', type=float)
        sort_by = request.args.get('sort_by', 'time')
        sort_order = request.args.get('sort_order', 'desc')
        page = max(1, request.args.get('page', 1, type=int))
        page_size = max(1, min(100, request.args.get('page_size', 20, type=int)))

        # --- Try SQLite first ---
        try:
            from .. import db as _db
            items, total, available_models, available_datasets = _db.query_eval_reports(
                search=search,
                models=models_filter,
                datasets=datasets_filter,
                score_min=score_min,
                score_max=score_max,
                sort_by=sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
            )
            # Sanity: filter out reports whose directories no longer exist
            items = [it for it in items if _report_dir_exists(it['name'], root)]
            # If filesystem check pruned items, re-count total
            if len(items) < page_size:
                all_total = _db.query_eval_reports(
                    search=search,
                    models=models_filter,
                    datasets=datasets_filter,
                    score_min=score_min,
                    score_max=score_max,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    page=1,
                    page_size=10000,
                )[0]
                all_filtered = [r for r in all_total if _report_dir_exists(r['name'], root)]
                total = len(all_filtered)
            return jsonify({
                'reports': items,
                'total': total,
                'page': page,
                'page_size': page_size,
                'filters': {
                    'available_models': available_models,
                    'available_datasets': available_datasets,
                },
            }), 200
        except Exception as db_err:
            logger.debug(f'SQLite query failed, falling back to filesystem: {db_err}')

        # --- Fallback: filesystem scan (original logic) ---
        raw_reports = scan_for_report_folders(root)
        items = []
        for rn in raw_reports:
            meta = _build_report_meta(rn, root)
            if meta is not None:
                items.append(meta)

        available_models = sorted({it['model_name'] for it in items})
        available_datasets = sorted({ds for it in items for ds in it['_datasets']})

        if search:
            items = [it for it in items if search in it['model_name'].lower() or search in it['dataset_name'].lower()]
        if models_filter:
            model_set = {m.strip().lower() for m in models_filter.split(';') if m.strip()}
            items = [it for it in items if it['model_name'].lower() in model_set]
        if datasets_filter:
            ds_set = {d.strip().lower() for d in datasets_filter.split(';') if d.strip()}
            items = [it for it in items if any(d.lower() in ds_set for d in it['_datasets'])]
        if score_min is not None:
            items = [it for it in items if it['score'] >= score_min]
        if score_max is not None:
            items = [it for it in items if it['score'] <= score_max]

        reverse = sort_order == 'desc'
        sort_key_map = {
            'score': lambda x: x['score'],
            'model': lambda x: x['model_name'].lower(),
            'dataset': lambda x: x['dataset_name'].lower(),
            'time': lambda x: x['timestamp'],
        }
        key_fn = sort_key_map.get(sort_by, sort_key_map['time'])
        items.sort(key=key_fn, reverse=reverse)

        total = len(items)
        start = (page - 1) * page_size
        page_items = items[start:start + page_size]

        for it in page_items:
            it.pop('_datasets', None)
            it.pop('_scores', None)

        return jsonify({
            'reports': page_items,
            'total': total,
            'page': page,
            'page_size': page_size,
            'filters': {
                'available_models': available_models,
                'available_datasets': available_datasets,
            },
        }), 200

    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to list reports: {e}', exc_info=True)
        return jsonify({'error': 'Failed to list reports', 'error_id': error_id}), 500


@bp_reports.route('/scan', methods=['GET'])
def scan_reports():
    """Scan the output directory for available report folders.

    Query params:
        root_path (str): directory to scan (default: OUTPUT_DIR)
    """
    try:
        root = _root_path()
        reports = scan_for_report_folders(root)
        return jsonify({'reports': reports}), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to scan reports: {e}', exc_info=True)
        return jsonify({'error': 'Failed to scan reports', 'error_id': error_id}), 500


@bp_reports.route('/load', methods=['GET'])
def load_report():
    """Load a single report by name.

    Query params:
        root_path   (str): output root directory
        report_name (str): report identifier
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    try:
        root = _root_path()
        validate_report_name(report_name, root)
        report_list, datasets, task_cfg = load_single_report(root, report_name)
        # Detect MTEB/RAG reports: results/ directory present but no standard report_list
        is_mteb = _is_mteb_report(root, report_name)
        if is_mteb:
            mteb_reports, mteb_datasets = _load_mteb_report_data(root, report_name, task_cfg)
            return jsonify({
                'report_list': mteb_reports,
                'datasets': mteb_datasets,
                'task_config': task_cfg,
                'eval_type': 'mteb',
            }), 200
        return jsonify({
            'report_list': [r.to_dict() for r in report_list],
            'datasets': datasets,
            'task_config': task_cfg,
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to load report {report_name}: {e}', exc_info=True)
        return jsonify({'error': 'Failed to load report', 'error_id': error_id}), 500


@bp_reports.route('/dataframe', methods=['GET'])
def get_dataframe():
    """Get report data as a flat JSON table.

    Query params:
        root_path        (str): output root directory
        report_name      (str): report identifier
        type             (str): 'acc' (accuracy overview) | 'compare' (pivot) | 'dataset' (single dataset)
        dataset_name     (str): required when type=dataset
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    df_type = request.args.get('type', 'acc')
    dataset_name = request.args.get('dataset_name', '')

    try:
        root = _root_path()
        validate_report_name(report_name, root)
        report_list, datasets, _ = load_single_report(root, report_name)
        acc_df, _ = get_acc_report_df(report_list)

        if df_type == 'compare':
            df, _ = get_compare_report_df(acc_df)
        elif df_type == 'dataset':
            if not dataset_name:
                return jsonify({'error': 'dataset_name is required for type=dataset'}), 400
            report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
            from evalscope.utils.data_utils import get_single_dataset_df
            df, _ = get_single_dataset_df(report_df, dataset_name)
        else:
            df = acc_df

        return jsonify({
            'columns': list(df.columns),
            'data': _df_to_records(df),
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get dataframe: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get dataframe', 'error_id': error_id}), 500


@bp_reports.route('/predictions', methods=['GET'])
def get_predictions():
    """Get model predictions for a given subset.

    Query params:
        root_path    (str): output root directory
        report_name  (str): report identifier
        dataset_name (str): dataset name
        subset_name  (str): subset name
    """
    report_name = request.args.get('report_name')
    dataset_name = request.args.get('dataset_name')
    subset_name = request.args.get('subset_name')

    if not all([report_name, dataset_name, subset_name]):
        return jsonify({'error': 'report_name, dataset_name and subset_name are required'}), 400

    try:
        root = _root_path()
        report_dir = validate_report_name(report_name, root)
        prefix, model_name, _ = process_report_name(report_name)
        work_dir = report_dir
        df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
        return jsonify({
            'predictions': _df_to_records(df),
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get predictions: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get predictions', 'error_id': error_id}), 500


@bp_reports.route('/analysis', methods=['GET'])
def get_analysis():
    """Get the AI analysis text for a dataset.

    Query params:
        root_path    (str): output root directory
        report_name  (str): report identifier
        dataset_name (str): dataset name
    """
    report_name = request.args.get('report_name')
    dataset_name = request.args.get('dataset_name')

    if not report_name or not dataset_name:
        return jsonify({'error': 'report_name and dataset_name are required'}), 400

    try:
        root = _root_path()
        validate_report_name(report_name, root)
        report_list, _, _ = load_single_report(root, report_name)
        analysis = get_report_analysis(report_list, dataset_name)
        return jsonify({'analysis': analysis}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get analysis: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get analysis', 'error_id': error_id}), 500


@bp_reports.route('/delete', methods=['DELETE'])
def delete_report():
    """Delete a report folder.

    Body (JSON):
        root_path   (str): output root directory
        report_name (str): report identifier
    """
    data = request.get_json(silent=True) or {}
    report_name = data.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400
    try:
        root = _root_path()
        prefix, _, _ = process_report_name(report_name)
        report_dir = os.path.realpath(os.path.join(root, prefix))
        # Security: ensure resolved path is within root
        if report_dir != root and not report_dir.startswith(root + os.sep):
            return jsonify({'error': 'Access denied: invalid report path'}), 403
        if not os.path.isdir(report_dir):
            return jsonify({'error': 'Report folder not found'}), 404
        import shutil
        shutil.rmtree(report_dir)
        logger.info(f'Deleted report: {report_dir}')

        # Sync SQLite
        task_id, _, _ = process_report_name(report_name)
        try:
            from .. import db as _db

            # Extract task_id (prefix) from composite report_name
            _db.delete_eval_report(task_id)
            logger.info(f'Deleted from SQLite: {task_id}')
        except Exception as e:
            logger.warning(f'Failed to delete {task_id} from SQLite: {e}')

        return jsonify({'ok': True}), 200
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to delete report {report_name}: {e}', exc_info=True)
        return jsonify({'error': 'Failed to delete report', 'error_id': error_id}), 500


@bp_reports.route('/html', methods=['GET'])
def get_html_report():
    """Serve the HTML report file for a given report.

    Query params:
        root_path   (str): output root directory
        report_name (str): report identifier
    """
    report_name = request.args.get('report_name')
    if not report_name:
        return jsonify({'error': 'report_name is required'}), 400

    try:
        root = os.path.abspath(_root_path())
        prefix, model_name, _ = process_report_name(report_name)
        report_html = os.path.realpath(os.path.join(root, prefix, OutputsStructure.REPORTS_DIR, 'report.html'))
        # Security: ensure resolved path is within root
        if not report_html.startswith(root + os.sep):
            return jsonify({'error': 'Access denied: invalid report path'}), 403

        if not os.path.exists(report_html):
            return jsonify({
                'error': 'Report not yet generated',
                'message': 'The HTML report has not been generated for this evaluation. It may still be in progress.',
            }), 404

        return send_file(report_html, mimetype='text/html')
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to get HTML report: {e}', exc_info=True)
        return jsonify({'error': 'Failed to get HTML report', 'error_id': error_id}), 500


@bp_reports.route('/chart', methods=['GET'])
def get_chart():
    """Generate an interactive Plotly chart as standalone HTML.

    Query params:
        root_path    (str): output root directory
        report_name  (str): report identifier (single report)
        report_names (str): semicolon-separated report identifiers (multi report)
        chart_type   (str): 'scores' | 'sunburst' | 'dataset_scores' | 'radar' | 'histogram' | 'grouped_bar'
        dataset_name (str): required for chart_type=dataset_scores
        subset_name  (str): required for chart_type=histogram
    """
    chart_type = request.args.get('chart_type', 'scores')
    root = _root_path()

    try:
        fig = None

        if chart_type == 'radar':
            names_raw = request.args.get('report_names', '')
            names = [n.strip() for n in names_raw.split(';') if n.strip()]
            if not names:
                # Fall back to singular report_name
                single = request.args.get('report_name', '').strip()
                if single:
                    names = [single]
                else:
                    return jsonify({'error': 'report_names or report_name is required for radar'}), 400
            for name in names:
                validate_report_name(name, root)
            report_list = load_multi_report(root, names)
            acc_df, _ = get_acc_report_df(report_list)
            fig = plot_multi_report_radar(acc_df)
        elif chart_type == 'grouped_bar':
            # Grouped bar chart for multi-model comparison
            names_raw = request.args.get('report_names', '')
            names = [n.strip() for n in names_raw.split(';') if n.strip()]
            if not names:
                return jsonify({'error': 'report_names is required for grouped_bar'}), 400
            for name in names:
                validate_report_name(name, root)
            report_list = load_multi_report(root, names)
            acc_df, _ = get_acc_report_df(report_list)
            color_seq = ['#816DF8', '#0F9C7E', '#fbbf24', '#a78bfa', '#63b3ed']
            fig = px.bar(
                acc_df,
                x=ReportKey.model_name,
                y=ReportKey.score,
                color=ReportKey.dataset_name,
                barmode='group',
                text=ReportKey.score,
                color_discrete_sequence=color_seq,
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(
                template=PLOTLY_THEME,
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                yaxis=dict(range=[0, 1]),
                margin=dict(t=20, l=20, r=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
        elif chart_type == 'histogram':
            # Score distribution histogram from prediction NScore values
            report_name = request.args.get('report_name')
            dataset_name = request.args.get('dataset_name', '')
            subset_name = request.args.get('subset_name', '')
            if not report_name or not dataset_name or not subset_name:
                return jsonify({'error': 'report_name, dataset_name and subset_name are required for histogram'}), 400
            report_dir = validate_report_name(report_name, root)
            prefix, model_name, _ = process_report_name(report_name)
            work_dir = report_dir
            pred_df = get_model_prediction(work_dir, model_name, dataset_name, subset_name)
            if pred_df is not None and not pred_df.empty and 'NScore' in pred_df.columns:
                fig = px.histogram(
                    pred_df,
                    x='NScore',
                    nbins=20,
                    color_discrete_sequence=['#816DF8'],
                )
                fig.update_layout(
                    template=PLOTLY_THEME,
                    xaxis_title='Score',
                    yaxis_title='Count',
                    margin=dict(t=20, l=20, r=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
        else:
            report_name = request.args.get('report_name')
            if not report_name:
                return jsonify({'error': 'report_name is required'}), 400
            validate_report_name(report_name, root)
            report_list, datasets, _ = load_single_report(root, report_name)
            acc_df, _ = get_acc_report_df(report_list)

            if chart_type == 'sunburst':
                fig = plot_single_report_sunburst(report_list)
            elif chart_type == 'dataset_scores':
                dataset_name = request.args.get('dataset_name', '')
                if not dataset_name:
                    return jsonify({'error': 'dataset_name is required for dataset_scores'}), 400
                report_df = get_data_frame(report_list=report_list, flatten_metrics=True, flatten_categories=True)
                from evalscope.utils.data_utils import get_single_dataset_df
                ds_df, _ = get_single_dataset_df(report_df, dataset_name)
                fig = plot_single_dataset_scores(ds_df)
            else:
                fig = plot_single_report_scores(acc_df)

        if fig is None:
            return '<html><body style="background:#0f172a;color:#94a3b8;display:flex;align-items:center;' \
                   'justify-content:center;height:100vh;font-family:sans-serif;">No data to plot</body></html>', \
                   200, {'Content-Type': 'text/html'}

        html = fig.to_html(full_html=True, include_plotlyjs=False, config={'responsive': True})
        plotly_script = f'<script src="{PLOTLY_CDN_URL}" charset="utf-8"></script>'
        html = html.replace('</head>', f'  {plotly_script}\n</head>')
        return html, 200, {'Content-Type': 'text/html'}

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.error(f'[{error_id}] Failed to generate chart: {e}', exc_info=True)
        return jsonify({'error': 'Failed to generate chart', 'error_id': error_id}), 500
