"""Dataset policy regressions without workers, downloads or provider requests."""
import copy

import pytest
from flask import Flask

from evalscope.config import TaskConfig
from evalscope.constants import EvalBackend
from evalscope.service.blueprints import eval as routes
from evalscope.service.blueprints.eval import RequestValidationError


@pytest.mark.parametrize('tool', ['mteb', 'MTEB'])
def test_common_policy_covers_batch_and_resumed_rag_configs(tool):
    data = {'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': {
        'tool': tool,
        'models': [{'model_name': 'audit', 'api_base': 'https://example.invalid/v1'}],
        'eval': {'offline': False, 'allow_download': True, 'limits': 2},
    }}
    original = copy.deepcopy(data)
    with pytest.raises(RequestValidationError, match='发布'):
        routes._apply_eval_dataset_policy(data, False)
    assert data == original


@pytest.mark.parametrize('backend', [EvalBackend.AIGC_EVAL, EvalBackend.AUDIO_EVAL])
def test_other_modules_are_unchanged(backend):
    data = {'eval_backend': backend, 'eval_config': {'custom': 'preserve'}}
    assert routes._apply_eval_dataset_policy(data, False) == data


@pytest.mark.parametrize('config', ['mteb.yaml', {'tool': None}, {'tool': 1}, {'tool': 'mteb', 'eval': []}])
def test_normal_user_malformed_rag_config_is_rejected(config):
    with pytest.raises(RequestValidationError):
        routes._apply_eval_dataset_policy({'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': config}, False)


@pytest.mark.parametrize('tool', ['ragas', 'clip_benchmark'])
def test_hidden_rag_tools_cannot_bypass_normal_user_data_policy(tool):
    data = {'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': {'tool': tool}}
    with pytest.raises(RequestValidationError):
        routes._apply_eval_dataset_policy(data, False)
    assert routes._apply_eval_dataset_policy(data, True) == data


@pytest.mark.parametrize('endpoint', ['/invoke', '/launch'])
@pytest.mark.parametrize('role', ['user', 'admin'])
def test_single_routes_pass_role_policy_to_final_task_config(endpoint, role, monkeypatch, tmp_path):
    from evalscope.service.blueprints import auth
    app = Flask(__name__)
    app.register_blueprint(routes.bp_eval, url_prefix='/eval')
    data = {'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': {
        'tool': 'MTEB',
        'models': [{'model_name': 'audit', 'api_base': 'https://example.invalid/v1'}],
        'eval': {'offline': False, 'allow_download': True},
    }}
    monkeypatch.setattr(routes, '_parse_request', lambda: (copy.deepcopy(data), 'eval_audit'))
    monkeypatch.setattr(auth, 'get_current_role', lambda: role)
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: 1)
    monkeypatch.setattr(routes, 'try_reserve_new_slot', lambda *a, **k: 'reserved')
    monkeypatch.setattr(routes, 'unregister_process', lambda *a, **k: None)
    monkeypatch.setattr(routes, 'OUTPUT_DIR', str(tmp_path))
    captured = []

    def execute(task_id, config, **kwargs):
        captured.append(config)
        return {'status': 'completed'}

    monkeypatch.setattr(routes, '_execute_task', execute)
    # Run background targets synchronously, without launching real workers.
    import threading
    class InlineThread:
        def __init__(self, target, args=(), kwargs=None, **ignored):
            self.target, self.args, self.kwargs = target, args, kwargs or {}
        def start(self):
            self.target(*self.args, **self.kwargs)
    monkeypatch.setattr(threading, 'Thread', InlineThread)
    response = app.test_client().post('/eval' + endpoint, json=data)
    if role == 'user':
        assert response.status_code == 400, response.get_json()
        assert '发布' in response.get_json()['error']
        assert not captured
        return
    assert response.status_code == (202 if endpoint == '/launch' else 200), response.get_json()
    assert len(captured) == 1
    args = captured[0].eval_config.eval
    assert args.offline is (role != 'admin')
    assert args.allow_download is (role == 'admin')


@pytest.mark.parametrize('endpoint', ['/invoke', '/launch', '/batch/launch', '/batch/resume'])
def test_hidden_rag_tools_rejected_at_http_boundary(endpoint, monkeypatch):
    from evalscope.service.blueprints import auth
    app = Flask(__name__)
    app.register_blueprint(routes.bp_eval, url_prefix='/eval')
    monkeypatch.setattr(auth, 'get_current_role', lambda: 'user')
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: 1)
    monkeypatch.setattr(routes, 'try_reserve_new_slot', lambda *a, **k: 'reserved')
    monkeypatch.setattr(routes, 'unregister_process', lambda *a, **k: None)
    response = app.test_client().post(
        '/eval' + endpoint,
        headers={'EvalScope-Task-Id': 'eval_audit'},
        json={'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': {'tool': 'RAGAS'},
              'batch_id': 'eval_batch_audit', 'upload_id': 'eval_upload_audit'},
    )
    assert response.status_code == 403, response.get_json()
    assert '普通用户' in response.get_json()['error']
