"""Endpoint regressions for destructive and downloadable task artifacts."""

from pathlib import Path

import pytest
from flask import Flask

import evalscope.service.blueprints.aigc as aigc
import evalscope.service.blueprints.audio as audio
import evalscope.service.blueprints.auth as auth
import evalscope.service.blueprints.perf as perf
import evalscope.service.blueprints.reports as reports
import evalscope.service.db as db


@pytest.fixture()
def app():
    return Flask(__name__)


def _response_status(result):
    return result[1] if isinstance(result, tuple) else result.status_code


@pytest.mark.parametrize(('module', 'view', 'url'), [
    (aigc, aigc.delete_aigc_report, '/api/v1/aigc/reports/.'),
    (audio, audio.delete_audio_report, '/api/v1/audio/reports/.'),
])
def test_delete_dot_never_removes_output_root(tmp_path, monkeypatch, app, module, view, url):
    root = tmp_path / module.__name__.rsplit('.', 1)[-1]
    root.mkdir()
    sentinel = root / 'keep.txt'
    sentinel.write_text('keep')
    monkeypatch.setattr(module, 'OUTPUT_DIR', root)
    monkeypatch.setattr(auth, 'check_task_ownership', lambda *_args, **_kwargs: (True, None))
    monkeypatch.setattr(db, 'delete_eval_report', lambda *_args, **_kwargs: None)

    with app.test_request_context(url, method='DELETE'):
        result = view('.')

    assert _response_status(result) == 400
    assert sentinel.read_text() == 'keep'


def test_perf_delete_dot_never_removes_output_root(tmp_path, monkeypatch, app):
    root = tmp_path / 'perf'
    root.mkdir()
    sentinel = root / 'keep.txt'
    sentinel.write_text('keep')
    monkeypatch.setattr(perf, 'OUTPUT_DIR', str(root))
    monkeypatch.setattr(auth, 'check_task_ownership', lambda *_args, **_kwargs: (True, None))
    monkeypatch.setattr(db, 'delete_perf_task', lambda *_args, **_kwargs: None)

    with app.test_request_context('/api/v1/perf/delete', method='DELETE', json={'task_id': '.'}):
        result = perf.delete_performance_test()

    assert _response_status(result) == 400
    assert sentinel.read_text() == 'keep'


@pytest.mark.parametrize(('module', 'view', 'allowed_dir', 'allowed_name'), [
    (aigc, aigc.serve_file, 'images', 'sample.png'),
    (audio, audio.serve_file, 'audio', 'sample.wav'),
])
def test_file_endpoint_only_serves_declared_artifacts(
    tmp_path, monkeypatch, app, module, view, allowed_dir, allowed_name
):
    root = tmp_path / module.__name__.rsplit('.', 1)[-1]
    task_dir = root / 'task_01'
    artifact = task_dir / allowed_dir / allowed_name
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'artifact')
    (task_dir / '.owner').write_text('1')
    (task_dir / 'task_config.yaml').write_text('private')
    logs = task_dir / 'logs'
    logs.mkdir()
    (logs / 'run.log').write_text('private log')

    monkeypatch.setattr(module, 'OUTPUT_DIR', root)
    monkeypatch.setattr(auth, 'check_task_artifact_access', lambda *_args, **_kwargs: True)

    with app.test_request_context(f'/file/task_01/{allowed_dir}/{allowed_name}'):
        response = view('task_01', f'{allowed_dir}/{allowed_name}')
        assert response.status_code == 200
        response.close()

    for private_name in ('.owner', 'task_config.yaml', 'logs/run.log'):
        with app.test_request_context(f'/file/task_01/{private_name}'):
            result = view('task_01', private_name)
            assert _response_status(result) == 403

    with app.test_request_context(f'/file/task_01/{allowed_dir}/missing.bin'):
        result = view('task_01', f'{allowed_dir}/missing.bin')
        assert _response_status(result) == 404


def test_report_delete_dot_never_removes_output_root(tmp_path, monkeypatch, app):
    root = tmp_path / 'reports'
    root.mkdir()
    sentinel = root / 'keep.txt'
    sentinel.write_text('keep')
    monkeypatch.setattr(reports, '_root_path', lambda: str(root))
    monkeypatch.setattr(reports, 'process_report_name', lambda _name: ('.', '', ''))

    with app.test_request_context('/api/v1/reports/delete', method='DELETE', json={'report_name': 'dot'}):
        result = reports.delete_report()

    assert _response_status(result) in (400, 403)
    assert sentinel.read_text() == 'keep'


@pytest.mark.parametrize(('view', 'subdir', 'filename'), [
    (aigc.serve_media, 'media', 'escape.png'),
    (aigc.serve_thumbnail, 'thumbnails', 'escape.jpg'),
])
def test_specialized_aigc_file_endpoints_reject_symlink_escape(
    tmp_path, monkeypatch, app, view, subdir, filename
):
    root = tmp_path / 'aigc'
    artifact_dir = root / 'task_01' / subdir
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / filename
    outside.write_bytes(b'private')
    (artifact_dir / filename).symlink_to(outside)

    monkeypatch.setattr(aigc, 'OUTPUT_DIR', root)
    monkeypatch.setattr(auth, 'check_task_artifact_access', lambda *_args, **_kwargs: True)

    with app.test_request_context(f'/{subdir}/task_01/{filename}'):
        result = view('task_01', filename)
    assert _response_status(result) == 403
