"""RAG radar uses results data rather than an absent reports directory."""
import json

from flask import Flask

from evalscope.service.blueprints import reports


def test_rag_radar_reads_mteb_results(monkeypatch, tmp_path):
    name = 'eval_radar@@embedding::STSBenchmark'
    task = tmp_path / 'eval_radar'
    result = task / 'results' / 'eval__embedding' / 'master'
    result.mkdir(parents=True)
    (result / 'STSBenchmark.json').write_text(json.dumps({
        'task_name': 'STSBenchmark', 'scores': {'test': [{'main_score': 0.75}]},
    }))
    configs = task / 'configs'
    configs.mkdir()
    (configs / 'task.yaml').write_text('eval_config:\n  eval:\n    limits: 2\n')
    app = Flask(__name__)
    app.register_blueprint(reports.bp_reports)
    monkeypatch.setattr(reports, '_root_path', lambda: str(tmp_path))
    monkeypatch.setattr(reports, 'validate_report_name', lambda *args: None)
    monkeypatch.setattr(reports, '_report_access_allowed', lambda *args: True)
    response = app.test_client().get('/api/v1/reports/chart', query_string={
        'report_name': name, 'chart_type': 'radar',
    })
    assert response.status_code == 200, response.get_json()
    assert b'STSBenchmark' in response.data
    assert b'0.75' in response.data
    assert b'scatterpolar' in response.data
