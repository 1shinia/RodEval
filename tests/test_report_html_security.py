"""Security regression tests for interactive HTML reports."""

import hashlib
from pathlib import Path

from flask import Flask
import plotly.graph_objects as go

from evalscope.constants import PLOTLY_CDN_URL, PLOTLY_LOCAL_URL
from evalscope.report.renderer import _md_to_html


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_markdown_report_content_removes_active_html():
    html = _md_to_html(
        '# Safe heading\n\n'
        '<script>alert(1)</script>\n\n'
        '<img src=x onerror="alert(2)">\n\n'
        '[bad](javascript:alert(3))\n\n'
        '<svg onload="alert(4)"></svg>\n\n'
        '| A | B |\n|---|---|\n| 1 | 2 |'
    )

    lowered = html.lower()
    assert '<script' not in lowered
    assert 'onerror' not in lowered
    assert 'onload' not in lowered
    assert 'javascript:' not in lowered
    assert '<svg' not in lowered
    assert '<h1>safe heading</h1>' in lowered
    assert '<table>' in lowered


def test_report_viewer_does_not_grant_same_origin_to_report_scripts():
    source = (
        REPO_ROOT / 'evalscope/web/src/pages/ReportViewerPage.tsx'
    ).read_text(encoding='utf-8')

    assert 'sandbox="allow-scripts"' in source
    assert 'allow-same-origin' not in source


def test_report_csp_sandboxes_scripts_without_same_origin():
    from evalscope.service.html_security import REPORT_CONTENT_SECURITY_POLICY

    directives = {
        part.strip()
        for part in REPORT_CONTENT_SECURITY_POLICY.split(';')
        if part.strip()
    }
    assert 'sandbox allow-scripts' in directives
    assert all('allow-same-origin' not in directive for directive in directives)
    assert "default-src 'none'" in directives
    assert any(directive.startswith('script-src ') for directive in directives)
    script_src = next(directive for directive in directives if directive.startswith('script-src '))
    assert PLOTLY_CDN_URL in script_src


def test_plotly_chart_iframe_does_not_grant_same_origin():
    source = (
        REPO_ROOT / 'evalscope/web/src/components/charts/PlotlyChart.tsx'
    ).read_text(encoding='utf-8')

    assert 'sandbox="allow-scripts"' in source
    assert 'allow-same-origin' not in source


def test_chart_endpoint_applies_report_sandbox(monkeypatch, tmp_path):
    from evalscope.service.blueprints import reports

    app = Flask(__name__)
    app.register_blueprint(reports.bp_reports)
    figure = go.Figure(go.Bar(x=['safe'], y=[1]))
    monkeypatch.setattr(reports, '_root_path', lambda: str(tmp_path))
    monkeypatch.setattr(reports, 'validate_report_name', lambda name, root: str(tmp_path / name))
    monkeypatch.setattr(reports, '_report_access_allowed', lambda name: True)
    monkeypatch.setattr(reports, 'load_single_report', lambda root, name: ([], [], None))
    monkeypatch.setattr(reports, 'get_acc_report_df', lambda report_list: (None, None))
    monkeypatch.setattr(reports, 'plot_single_report_scores', lambda frame: figure)

    response = app.test_client().get('/api/v1/reports/chart?report_name=report-a')

    assert response.status_code == 200
    assert response.headers['Content-Security-Policy'].endswith('sandbox allow-scripts')
    assert 'allow-same-origin' not in response.headers['Content-Security-Policy']
    local_url = f'http://localhost{PLOTLY_LOCAL_URL}'
    assert PLOTLY_LOCAL_URL.encode() in response.data
    assert local_url in response.headers['Content-Security-Policy']
    assert PLOTLY_CDN_URL.encode() not in response.data


def test_html_report_uses_local_plotly_but_download_remains_portable(monkeypatch, tmp_path):
    from evalscope.service.blueprints import reports

    report_dir = tmp_path / 'eval_test' / 'reports'
    report_dir.mkdir(parents=True)
    report_html = report_dir / 'report.html'
    report_html.write_text(f'<script src="{PLOTLY_CDN_URL}"></script>', encoding='utf-8')

    app = Flask(__name__)
    app.register_blueprint(reports.bp_reports)
    monkeypatch.setattr(reports, '_root_path', lambda: str(tmp_path))
    monkeypatch.setattr(reports, '_report_access_allowed', lambda name: True)
    monkeypatch.setattr(reports, 'process_report_name', lambda name: ('eval_test', 'model', ['dataset']))

    client = app.test_client()
    inline = client.get('/api/v1/reports/html?report_name=report-a')
    download = client.get('/api/v1/reports/html?report_name=report-a&download=1')

    local_url = f'http://localhost{PLOTLY_LOCAL_URL}'
    assert inline.status_code == 200
    assert PLOTLY_LOCAL_URL.encode() in inline.data
    assert PLOTLY_CDN_URL.encode() not in inline.data
    assert local_url in inline.headers['Content-Security-Policy']
    assert download.status_code == 200
    assert PLOTLY_CDN_URL.encode() in download.data


def test_plotly_csp_uses_forwarded_origin_only_from_trusted_proxy(monkeypatch):
    from evalscope.service.blueprints import reports

    app = Flask(__name__)
    app.register_blueprint(reports.bp_reports)
    headers = {'X-Forwarded-Host': 'public.example:5173', 'X-Forwarded-Proto': 'https'}

    with app.test_request_context('/', headers=headers, environ_base={'REMOTE_ADDR': '127.0.0.1'}):
        trusted_source = reports._local_plotly_source()
    with app.test_request_context('/', headers=headers, environ_base={'REMOTE_ADDR': '203.0.113.10'}):
        untrusted_source = reports._local_plotly_source()

    assert trusted_source == f'https://public.example:5173{PLOTLY_LOCAL_URL}'
    assert untrusted_source == f'http://localhost{PLOTLY_LOCAL_URL}'


def test_local_plotly_asset_is_versioned_and_immutable():
    from evalscope.service.blueprints import reports

    app = Flask(__name__)
    app.register_blueprint(reports.bp_reports)
    response = app.test_client().get(PLOTLY_LOCAL_URL)

    assert response.status_code == 200
    assert response.content_type == 'application/javascript; charset=utf-8'
    assert response.headers['Cache-Control'] == 'public, max-age=31536000, immutable'
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert len(response.data) == 4_558_696
    assert hashlib.sha256(response.data).hexdigest() == (
        '6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603'
    )


def test_plotly_encodes_script_closing_sequences_in_chart_data():
    payload = 'X</script><script>globalThis.PWN=1</script>'
    html = go.Figure(go.Bar(x=[payload], y=[1])).to_html(
        full_html=False,
        include_plotlyjs=False,
    )

    assert '</script><script>globalThis.PWN' not in html
    assert r'\u003c\u002fscript\u003e' in html
