"""Launch endpoint slot-release regression tests.

The /eval/launch and /perf/launch endpoints reserve a concurrency slot up
front (``try_reserve_new_slot``), then run several validation checks before
spawning the background thread.  Historically those early-return paths (400
missing config, 500 model-launch failure) never released the reserved slot,
so a few malformed requests would permanently exhaust the per-user concurrency
limit.

Regression guard: a request that fails validation *after* the slot is reserved
must leave the registry empty for that task_id (the outer ``finally`` releases
the slot on every path that did not hand off to the background thread).

Constraint: importing evalscope.service pulls in uvicorn/waitress through the
perf module chain, so this module must run under the hermes env
(``/root/anaconda3/envs/hermes/bin/python -m pytest``).
"""
import pytest

import evalscope.service.blueprints.eval as svc_eval
import evalscope.service.blueprints.perf as svc_perf
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log
from evalscope.service.utils import process


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils, svc_eval, svc_perf):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)

    monkeypatch.setenv('EVALSCOPE_ADMIN_PASSWORD', 'testpass')
    from evalscope.service.app import create_app
    app = create_app(outputs=root)

    c = app.test_client()
    resp = c.post('/api/v1/auth/register', json={'username': 'launch_slot_user', 'password': 'pw123456'})
    assert resp.status_code == 201, resp.data
    token = resp.get_json()['token']
    c.environ_base['HTTP_AUTHORIZATION'] = 'Bearer ' + token
    return c


def _running_ids():
    return [t['task_id'] for t in process.get_running_tasks()]


def test_eval_launch_rag_missing_config_releases_slot(client):
    """RAG eval without eval_config returns 400 *and* frees the reserved slot."""
    task_id = 'eval_launch_slot_rag'
    resp = client.post(
        '/api/v1/eval/launch',
        json={'eval_backend': 'RAGEval'},
        headers={'EvalScope-Task-Id': task_id},
    )
    assert resp.status_code == 400
    assert task_id not in _running_ids()


def test_perf_launch_missing_model_releases_slot(client):
    """Perf launch missing the required model field returns 400 and frees the slot."""
    task_id = 'perf_launch_slot_model'
    resp = client.post(
        '/api/v1/perf/launch',
        json={'api': 'openai'},
        headers={'EvalScope-Task-Id': task_id},
    )
    assert resp.status_code == 400
    assert task_id not in _running_ids()
