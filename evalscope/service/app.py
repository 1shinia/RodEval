# Copyright (c) Alibaba, Inc. and its affiliates.
"""Flask service for EvalScope evaluation and performance testing."""
import multiprocessing
import os
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

from evalscope.utils.logger import get_logger
from . import db as _db
from .blueprints import bp_eval, bp_perf, bp_reports
from .utils import OUTPUT_DIR as _DEFAULT_ROOT

logger = get_logger()

# Path to the built React SPA (web/dist).  Resolved relative to the
# repository root so that ``pip install -e .`` works out of the box.
_WEB_DIST = os.path.join(os.path.dirname(__file__), '..', 'web', 'dist')
_WEB_DIST = os.path.abspath(_WEB_DIST)


def create_app(outputs: str = None):
    """Create and configure the Flask application.

    Args:
        outputs: Root directory for evaluation outputs. If provided, it will be
                 used as the default scan path in the web dashboard.
    """
    app = Flask(__name__)

    # Store the outputs path in app config so blueprints can access it
    if outputs:
        app.config['OUTPUTS_ROOT'] = os.path.abspath(outputs)
    else:
        app.config['OUTPUTS_ROOT'] = None

    # Ensure non-ASCII characters (e.g. Chinese) are serialised as-is in JSON
    # responses instead of being escaped to \uXXXX sequences.
    app.json.ensure_ascii = False

    # --- Load .env from project root (if python-dotenv is installed) ------
    _env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    _env_path = os.path.abspath(_env_path)
    try:
        from dotenv import load_dotenv
        if os.path.isfile(_env_path):
            load_dotenv(_env_path, override=False)
    except ImportError:
        pass

    # --- CORS (restrict to known frontend origins) ----------------------
    try:
        from flask_cors import CORS

        # Default local-dev origins; additional origins can be added via
        # the CORS_ORIGINS environment variable (comma-separated) or
        # a .env file in the project root.
        # Example .env: CORS_ORIGINS=http://10.192.161.184:5173,http://my-server:80
        allowed_origins = [
            'http://localhost:5173',
            'http://127.0.0.1:5173',
        ]
        extra = os.environ.get('CORS_ORIGINS', '')
        if extra:
            allowed_origins.extend(o.strip() for o in extra.split(',') if o.strip())
        CORS(app, resources={r'/api/*': {'origins': allowed_origins}})
    except ImportError:
        pass  # flask-cors not installed; same-origin only

    # Register blueprints
    app.register_blueprint(bp_eval)
    app.register_blueprint(bp_perf)
    app.register_blueprint(bp_reports)

    # --- Initialise SQLite metadata store --------------------------------
    outputs_root = app.config.get('OUTPUTS_ROOT') or _DEFAULT_ROOT
    try:
        _db.init_db(outputs_root)
        _db.backfill(outputs_root)
        _db.recover_stale_tasks()
    except Exception as e:
        logger.warning(f'SQLite metadata store init failed (non-fatal): {e}')

    # --- Set up rotating file logging for the service --------------------
    try:
        from evalscope.utils.logger import setup_service_logging
        log_file = os.path.join(outputs_root, 'evalscope_service.log')
        setup_service_logging(log_file)
    except Exception as e:
        logger.debug(f'Service log rotation setup failed (non-fatal): {e}')

    # --- Clean up old task logs (retention policy) -----------------------
    try:
        from .utils.log import cleanup_old_task_logs
        result = cleanup_old_task_logs()
        if result.get('removed'):
            freed_mb = result['freed_bytes'] / (1024 * 1024)
            logger.info('Startup cleanup: removed %d old task dirs, freed %.1f MB', result['removed'], freed_mb)
    except Exception as e:
        logger.debug(f'Startup log cleanup failed (non-fatal): {e}')

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint with component status."""
        checks: dict = {'status': 'ok', 'service': 'evalscope', 'timestamp': datetime.now().isoformat()}

        # SQLite connectivity
        try:
            conn = _db._get_conn()
            conn.execute('SELECT 1')
            checks['sqlite'] = 'ok'
        except Exception as e:
            checks['sqlite'] = f'error: {e}'
            checks['status'] = 'degraded'

        # Running tasks count
        try:
            from .utils import get_running_tasks as _get_running_tasks
            checks['running_tasks'] = len(_get_running_tasks())
        except Exception:
            checks['running_tasks'] = -1

        # Disk space for output directory
        try:
            import shutil
            outputs_root = app.config.get('OUTPUTS_ROOT') or _DEFAULT_ROOT
            usage = shutil.disk_usage(outputs_root)
            checks['disk_free_gb'] = round(usage.free / (1024**3), 2)
            if usage.free < 1024 * 1024 * 100:  # < 100 MB
                checks['status'] = 'degraded'
        except Exception:
            pass

        status_code = 200 if checks['status'] == 'ok' else 503
        return jsonify(checks), status_code

    @app.route('/api/v1/config', methods=['GET'])
    def get_config():
        """Return runtime configuration for the frontend."""
        outputs_root = app.config.get('OUTPUTS_ROOT')
        # Auto-detect backend server address: get the actual server IP and port
        # Use the socket connection to find the real server address
        server_address = request.environ.get('HTTP_HOST', request.host)
        # If accessed through Vite proxy, the host will be the frontend address
        # We need to replace the port with the backend port
        if ':' in server_address:
            host_part = server_address.split(':')[0]
            backend_port = request.environ.get('SERVER_PORT', '9000')
            server_address = f'{host_part}:{backend_port}'
        return jsonify({
            'outputs_root': outputs_root or _DEFAULT_ROOT,
            'server_address': server_address,
        })

    @app.route('/api/v1/tasks/running', methods=['GET'])
    def get_running_tasks():
        """Return a list of currently running tasks with metadata.

        Merges in-memory registry with SQLite-persisted state so that
        tasks surviving a server restart are still visible.
        """
        from .utils import get_running_tasks as _get_running_tasks
        tasks = _get_running_tasks()
        seen_ids = {t['task_id'] for t in tasks}

        # Add SQLite-persisted tasks that are still marked 'running'
        try:
            db_tasks = _db.list_running_tasks()
            for dt in db_tasks:
                if dt['task_id'] not in seen_ids:
                    tasks.append({
                        'task_id': dt['task_id'],
                        'task_type': dt['task_type'],
                        'model': dt['model'],
                        'start_time': dt['started_at'],
                        'elapsed_seconds': None,
                        'source': 'sqlite',  # indicate this was recovered from DB
                    })
        except Exception as e:
            logger.debug(f'Failed to load SQLite running tasks: {e}')

        return jsonify({'tasks': tasks, 'count': len(tasks)})

    # --- SPA static-file serving ------------------------------------------
    if os.path.isdir(_WEB_DIST):

        # Pre-resolve the web dist directory for path-traversal guard.
        _web_dist_real = os.path.realpath(_WEB_DIST)

        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_spa(path):
            """Serve the React SPA for all non-API routes."""
            file_path = os.path.realpath(os.path.join(_WEB_DIST, path))
            # Prevent path traversal: the resolved file must stay inside _WEB_DIST.
            if not file_path.startswith(_web_dist_real + os.sep) and file_path != _web_dist_real:
                return jsonify({'error': 'Not found'}), 404
            if path and os.path.isfile(file_path):
                return send_from_directory(_WEB_DIST, path)
            return send_from_directory(_WEB_DIST, 'index.html')

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Endpoint not found',
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Internal server error: {str(error)}', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(Exception)
    def handle_unhandled(error):
        logger.error(f'Unhandled exception: {str(error)}', exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    return app


def run_service(host: str = '0.0.0.0', port: int = 9000, debug: bool = False, outputs: str = None, threads: int = 16):
    """Run the EvalScope service.

    Uses waitress (multi-threaded WSGI server) for proper concurrent request
    handling.  Falls back to Flask's built-in dev server if waitress is not
    installed or if ``debug=True``.

    Args:
        outputs: Root directory for evaluation outputs. If provided, the web
                 dashboard will use this as the default scan path instead of
                 ``./outputs``.
        threads: Number of worker threads for waitress (default: 4).
    """

    # Force the multiprocessing start method to 'spawn' to avoid issues with
    # model loading in forked child processes on some platforms.
    multiprocessing.set_start_method('spawn', force=True)
    app = create_app(outputs=outputs)

    logger.info('Available endpoints:')
    logger.info('  GET  /health                         - Health check')
    logger.info('  GET  /api/v1/config                  - Get runtime configuration')
    logger.info('  GET  /api/v1/tasks/running           - List currently running tasks')
    logger.info('  POST /api/v1/eval/invoke             - Run model evaluation task (blocking)')
    logger.info('  POST /api/v1/eval/resume/invoke      - Resume interrupted evaluation task')
    logger.info('  GET  /api/v1/eval/benchmarks         - List supported benchmarks with descriptions')
    logger.info('  GET  /api/v1/eval/log                - Get evaluation log')
    logger.info('  GET  /api/v1/eval/progress           - Get real-time evaluation progress')
    logger.info('  GET  /api/v1/eval/progress/stream    - SSE stream for eval progress')
    logger.info('  GET  /api/v1/eval/log/stream         - SSE stream for eval log')
    logger.info('  GET  /api/v1/eval/report             - Get HTML evaluation report')
    logger.info('  POST /api/v1/perf/invoke             - Run performance benchmark task (blocking)')
    logger.info('  GET  /api/v1/perf/log                - Get performance benchmark log')
    logger.info('  GET  /api/v1/perf/progress           - Get real-time performance benchmark progress')
    logger.info('  GET  /api/v1/perf/progress/stream    - SSE stream for perf progress')
    logger.info('  GET  /api/v1/perf/log/stream         - SSE stream for perf log')
    logger.info('  GET  /api/v1/perf/report             - Get HTML performance benchmark report')
    logger.info('Refer to docs for parameters: https://evalscope.readthedocs.io/en/latest/user_guides/service.html')

    # Print a user-friendly dashboard URL
    display_host = host if host != '0.0.0.0' else '127.0.0.1'
    dashboard_url = f'http://{display_host}:{port}/dashboard'
    logger.info(f'Dashboard: {dashboard_url}')
    print(f'\n  🌐 EvalScope Dashboard: {dashboard_url}\n')

    if debug:
        logger.info('Debug mode: using Flask dev server')
        app.run(host=host, port=port, debug=True)
    else:
        try:
            from waitress import serve
            logger.info(f'Starting with waitress on {host}:{port} (threads={threads})')
            serve(app, host=host, port=port, threads=threads, _quiet=False)
        except ImportError:
            logger.warning('waitress not installed, falling back to Flask dev server (single-threaded)')
            logger.warning('Install waitress for production use: pip install waitress')
            app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='EvalScope Service')
    parser.add_argument('--port', type=int, default=9000, help='Port to listen on (default: 9000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--outputs', type=str, default=None, help='Output directory')
    parser.add_argument('--threads', type=int, default=16, help='Worker threads')
    args = parser.parse_args()
    run_service(host=args.host, port=args.port, debug=args.debug, outputs=args.outputs, threads=args.threads)
