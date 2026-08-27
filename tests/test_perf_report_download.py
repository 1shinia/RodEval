"""Perf report download endpoint behaviour.

Covers the ``download=1`` attachment mode added to ``/api/v1/perf/report``
(alignment with eval reports ``/api/v1/reports/html``).

OUTPUT_DIR is module-level bound in three places, so all three are patched
to an isolated tmp dir; the app is built per-test on the same tmp root.
"""
import pytest

import evalscope.service.blueprints.perf as svc_perf
import evalscope.service.utils as svc_utils
import evalscope.service.utils.log as svc_log

TASK_ID = 'perf_dltest_001'
REPORT_BODY = '<html><body>perf report</body></html>'


@pytest.fixture()
def perf_report_client(tmp_path, monkeypatch):
    root = str(tmp_path)
    for mod in (svc_log, svc_utils, svc_perf):
        monkeypatch.setattr(mod, 'OUTPUT_DIR', root)

    report_dir = tmp_path / TASK_ID / 'perf'
    report_dir.mkdir(parents=True)
    (report_dir / 'perf_report.html').write_text(REPORT_BODY)

    from evalscope.service.app import create_app
    app = create_app(outputs=root)
    return app.test_client()


def test_inline_serving_unchanged(perf_report_client):
    resp = perf_report_client.get(f'/api/v1/perf/report?task_id={TASK_ID}')
    assert resp.status_code == 200
    assert 'text/html' in resp.content_type
    assert 'attachment' not in (resp.headers.get('Content-Disposition') or '')
    assert resp.data.decode() == REPORT_BODY


@pytest.mark.parametrize('value', ['1', 'true', 'yes'])
def test_download_attachment(perf_report_client, value):
    resp = perf_report_client.get(f'/api/v1/perf/report?task_id={TASK_ID}&download={value}')
    assert resp.status_code == 200
    cd = resp.headers.get('Content-Disposition') or ''
    assert 'attachment' in cd
    assert f'{TASK_ID}_perf_report.html' in cd
    assert resp.data.decode() == REPORT_BODY


def test_download_extra_query_param(perf_report_client):
    resp = perf_report_client.get(f'/api/v1/perf/report?task_id={TASK_ID}&download=1&x=1')
    assert resp.status_code == 200
    assert 'attachment' in (resp.headers.get('Content-Disposition') or '')


def test_missing_task_returns_404(perf_report_client):
    resp = perf_report_client.get('/api/v1/perf/report?task_id=no_such_task_xyz')
    assert resp.status_code == 404


def test_path_traversal_task_id_rejected(perf_report_client):
    resp = perf_report_client.get('/api/v1/perf/report?task_id=../../etc/passwd')
    assert resp.status_code == 400