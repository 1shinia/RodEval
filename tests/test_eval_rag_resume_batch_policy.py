"""Exercise real resume/batch route configuration paths in an isolated workspace."""
import threading
from unittest.mock import Mock

import pytest
from flask import Flask

from evalscope.config import TaskConfig
from evalscope.constants import EvalBackend
from evalscope.service.blueprints import auth
from evalscope.service.blueprints import eval as routes


class InlineThread:
    def __init__(self, target, args=(), kwargs=None, **ignored):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    app = Flask(__name__)
    app.register_blueprint(routes.bp_eval, url_prefix='/eval')
    monkeypatch.setattr(routes, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(auth, 'get_current_user_id', lambda: 1)
    monkeypatch.setattr(auth, 'check_task_artifact_access', lambda *a, **k: True)
    monkeypatch.setattr(routes, 'try_reserve_slot', lambda *a, **k: True)
    monkeypatch.setattr(routes, 'try_reserve_new_slot', lambda *a, **k: 'reserved')
    monkeypatch.setattr(routes, 'unregister_process', lambda *a, **k: None)
    monkeypatch.setattr(routes, 'create_log_file', lambda *a, **k: None)
    monkeypatch.setattr(threading, 'Thread', InlineThread)
    captured = []
    def execute(task_id, config, **kwargs):
        captured.append(config)
        return {'status': 'completed'}
    monkeypatch.setattr(routes, '_execute_task', execute)
    return app.test_client(), captured


def rag_config():
    return {'tool': 'MTEB', 'models': [
        {'model_name': 'audit', 'api_base': 'https://example.invalid/v1'}
    ], 'eval': {'offline': False, 'allow_download': True, 'limits': 2}}


@pytest.mark.parametrize('role', ['user', 'admin'])
def test_resume_reauthorizes_saved_online_config(role, harness, tmp_path, monkeypatch):
    client, captured = harness
    monkeypatch.setattr(auth, 'get_current_role', lambda: role)
    work = tmp_path / 'eval_resume_audit'
    work.mkdir()
    (work / 'task_config.yaml').write_text('placeholder')
    saved = TaskConfig(eval_backend=EvalBackend.RAG_EVAL, eval_config=rag_config())
    monkeypatch.setattr(TaskConfig, 'from_yaml', lambda *a, **k: saved)
    response = client.post('/eval/resume/invoke', json={'task_id': work.name})
    if role == 'user':
        assert response.status_code == 400, response.get_json()
        assert '发布' in response.get_json()['error']
        assert not captured
        return
    assert response.status_code == 200, response.get_json()
    assert len(captured) == 1
    args = captured[0].eval_config.eval
    assert args.offline is (role != 'admin')
    assert args.allow_download is (role == 'admin')


@pytest.mark.parametrize('role', ['user', 'admin'])
def test_resume_hidden_tool_rechecks_current_role(role, harness, tmp_path, monkeypatch):
    client, captured = harness
    monkeypatch.setattr(auth, 'get_current_role', lambda: role)
    work = tmp_path / 'eval_hidden_resume_audit'
    work.mkdir()
    (work / 'task_config.yaml').write_text('placeholder')
    saved = TaskConfig(eval_backend=EvalBackend.RAG_EVAL, eval_config={
        'tool': 'clip_benchmark', 'model': 'audit', 'dataset': 'audit',
    })
    monkeypatch.setattr(TaskConfig, 'from_yaml', lambda *a, **k: saved)
    if role == 'admin':
        # Other-tool dependencies are outside this regression; verify policy preserves config directly.
        assert routes._apply_eval_dataset_policy(saved.to_dict(), True)['eval_config']['tool'] == 'clip_benchmark'
        return
    response = client.post('/eval/resume/invoke', json={'task_id': work.name})
    assert response.status_code == 403, response.get_json()
    assert not captured


@pytest.mark.parametrize('role', ['user', 'admin'])
@pytest.mark.parametrize('resume', [False, True])
def test_batch_children_receive_role_policy(role, resume, harness, tmp_path, monkeypatch):
    client, captured = harness
    monkeypatch.setattr(auth, 'get_current_role', lambda: role)
    monkeypatch.setattr(routes, 'EVAL_BATCH_UPLOAD_DIR', str(tmp_path))
    monkeypatch.setattr(routes, '_eval_batch_state', {})
    batch_id = 'eval_batch_audit'
    upload_id = 'eval_upload_audit' if resume else batch_id
    (tmp_path / (upload_id + '.csv')).write_text('model,base_url\naudit,https://example.invalid/v1\n')
    (tmp_path / (upload_id + '.csv.owner')).write_text('1')
    db = Mock()
    db.get_batch_job.return_value = {
        'total': 1, 'completed': 0, 'errors': 0, 'results': [], 'error_details': [],
        'items': [{'row_index': 0, 'status': 'interrupted'}],
    }
    db.claim_batch_resume.return_value = 'claimed'
    db.BatchCheckpointError = type('CheckpointError', (Exception,), {})
    monkeypatch.setattr(routes, '_db', db)
    payload = {'batch_id': batch_id, 'eval_backend': EvalBackend.RAG_EVAL, 'eval_config': rag_config()}
    if resume:
        payload['upload_id'] = upload_id
    response = client.post('/eval/batch/' + ('resume' if resume else 'launch'), json=payload)
    if role == 'user':
        assert response.status_code == 400, response.get_json()
        assert '发布' in response.get_json()['error']
        assert not captured
        return
    assert response.status_code == 200, response.get_json()
    assert len(captured) == 1, routes._eval_batch_state
    args = captured[0].eval_config.eval
    assert args.offline is (role != 'admin')
    assert args.allow_download is (role == 'admin')
    assert args.limits == 2
