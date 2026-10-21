import json

import pytest
from flask import Flask

from evalscope.service.blueprints import reports


@pytest.fixture
def client(monkeypatch, tmp_path):
    result = tmp_path / 'eval_html' / 'results' / 'eval__reranker' / 'master'
    result.mkdir(parents=True)
    (result / 'AskUbuntuDupQuestions.json').write_text(json.dumps({
        'task_name': 'AskUbuntuDupQuestions',
        'scores': {'test': [{'main_score': 0.3077, 'ndcg_at_10': 0.42,
                            'hf_subset': '<script>alert(1)</script>'}]},
    }))
    configs = tmp_path / 'eval_html' / 'configs'
    configs.mkdir()
    (configs / 'task.yaml').write_text('eval_config:\n  eval:\n    limits: 2\n')
    app = Flask(__name__)
    app.register_blueprint(reports.bp_reports)
    monkeypatch.setattr(reports, '_root_path', lambda: str(tmp_path))
    monkeypatch.setattr(reports, '_report_access_allowed', lambda *args: True)
    return app.test_client()


def test_completed_rag_html_without_generated_file(client):
    response = client.get('/api/v1/reports/html', query_string={
        'report_name': 'eval_html@@reranker::AskUbuntuDupQuestions',
    })
    assert response.status_code == 200
    assert response.mimetype == 'text/html'
    assert b'AskUbuntuDupQuestions' in response.data
    assert b'0.3077' in response.data
    assert b'ndcg_at_10' in response.data
    assert b'<script>alert(1)</script>' not in response.data


def test_rag_download(client):
    response = client.get('/api/v1/reports/html', query_string={
        'report_name': 'eval_html@@reranker::AskUbuntuDupQuestions', 'download': '1',
    })
    assert response.status_code == 200
    assert 'attachment' in response.headers['Content-Disposition']


def test_rag_html_keeps_access_check(client, monkeypatch):
    monkeypatch.setattr(reports, '_report_access_allowed', lambda *args: False)
    response = client.get('/api/v1/reports/html?report_name=eval_html@@reranker::AskUbuntuDupQuestions')
    assert response.status_code == 404
    assert b'0.3077' not in response.data
