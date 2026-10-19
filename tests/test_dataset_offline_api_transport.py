"""Dataset offline mode must not disable model API HTTP transport."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import requests

from evalscope.config import TaskConfig
from evalscope.service.utils import process


def test_dataset_offline_worker_can_call_local_api(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'local-transport-ok')

        def log_message(self, format, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def run(config):
            return requests.post(
                f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                json={}, timeout=5,
            ).text

        monkeypatch.setattr(process, 'run_task', run)
        config = TaskConfig(model='audit', eval_type='openai_api', dataset_offline=True)
        assert process.run_eval_wrapper(config) == 'local-transport-ok'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
