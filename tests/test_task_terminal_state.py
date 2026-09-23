import json

import pytest

from evalscope.service import progress_state
from evalscope.service.blueprints import perf


def test_terminal_progress_preserves_counters_and_writes_canonical_failure(tmp_path):
    progress_file = tmp_path / 'progress.json'
    progress_file.write_text(json.dumps({'processed': 3, 'total': 10}), encoding='utf-8')

    progress_state.write_terminal_progress(
        str(progress_file), 'failed', pipeline='perf', error='report missing'
    )

    data = json.loads(progress_file.read_text(encoding='utf-8'))
    assert data['status'] == 'failed'
    assert data['phase'] == 'failed'
    assert data['processed'] == 3
    assert data['total'] == 10
    assert data['error'] == 'report missing'
    assert data['updated_at'].endswith('+00:00')


def test_stopped_progress_cannot_be_overwritten_by_late_failure(tmp_path):
    progress_file = tmp_path / 'progress.json'
    progress_file.write_text(json.dumps({'status': 'stopped', 'percent': 42}), encoding='utf-8')

    progress_state.write_terminal_progress(
        str(progress_file), 'failed', pipeline='perf', error='process terminated'
    )

    data = json.loads(progress_file.read_text(encoding='utf-8'))
    assert data == {'status': 'stopped', 'percent': 42}


def test_shared_terminal_statuses_include_partial_and_orphaned():
    assert 'partial_success' in perf._TERMINAL_PROGRESS_STATUSES
    assert 'orphaned' in perf._TERMINAL_PROGRESS_STATUSES


def test_perf_completion_requires_a_real_report(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, 'OUTPUT_DIR', str(tmp_path))
    task_id = 'perf_report_missing'
    (tmp_path / task_id / 'perf').mkdir(parents=True)

    with pytest.raises(RuntimeError, match='without generating a report'):
        perf._require_perf_report(task_id)

    (tmp_path / task_id / 'perf' / 'perf_report.html').write_text('<html/>', encoding='utf-8')
    perf._require_perf_report(task_id)


def test_sla_summary_is_a_valid_perf_report(monkeypatch, tmp_path):
    monkeypatch.setattr(perf, 'OUTPUT_DIR', str(tmp_path))
    task_id = 'perf_sla_report'
    (tmp_path / task_id).mkdir(parents=True)
    (tmp_path / task_id / 'sla_summary.json').write_text('{}', encoding='utf-8')

    perf._require_perf_report(task_id)
