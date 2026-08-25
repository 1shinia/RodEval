"""Regression tests for AIGC upsert failure-flagging (config-driven).

The original logic hardcoded clip_score_mean as THE primary metric, which
mis-flagged img2img/txt2img runs configured with lpips only (and txt2video
runs configured with fvd only, where fvd was actually produced). Expected
metrics must come from eval_config.eval.metrics in the task config.
"""
import json
import os
import sqlite3
import tempfile

import pytest

from evalscope.service import db as _db


@pytest.fixture()
def iso_db():
    tmp = tempfile.mkdtemp()
    _db._local.conn = None
    _db.init_db(tmp)
    yield tmp
    _db._db_path = None
    _db._local.conn = None


def _make_task(tmp: str, task_id: str, model_type: str, metrics: dict, expected: list[str]) -> str:
    """Create task_dir with results.json + configs/task_config.yaml. Returns task dir."""
    task_dir = os.path.join(tmp, task_id)
    os.makedirs(os.path.join(task_dir, 'configs'))
    with open(os.path.join(task_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'm', 'model_type': model_type, 'tool': 'txt2video',
            'num_samples': 1, 'timestamp': 0, 'metrics': metrics,
        }, f)
    with open(os.path.join(task_dir, 'configs', 'task_config.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"eval_config:\n  eval:\n    metrics: {json.dumps(expected)}\n")
    return task_dir


def _flag(iso_db: str, task_id: str):
    con = sqlite3.connect(os.path.join(iso_db, 'evalscope_meta.db'))
    row = con.execute('SELECT has_errors, error_note FROM eval_reports WHERE task_id = ?', (task_id,)).fetchone()
    con.close()
    return row


def test_lpips_config_not_flagged(iso_db):
    """img2img/txt2img configured with lpips must NOT be flagged when lpips is present."""
    _make_task(iso_db, 't_lpips', 'img2img', {'lpips_mean': 0.0012}, ['lpips'])
    assert _db.upsert_aigc_audio_report(iso_db, 't_lpips') is True
    assert _flag(iso_db, 't_lpips') == (0, '')
    con = sqlite3.connect(os.path.join(iso_db, 'evalscope_meta.db'))
    score = con.execute('SELECT score FROM eval_reports WHERE task_id = ?', ('t_lpips',)).fetchone()[0]
    con.close()
    assert score == pytest.approx(0.0012)  # primary score follows the configured metric


def test_missing_clip_score_flagged(iso_db):
    """txt2video configured with clip_score but no frames -> metrics empty -> flagged."""
    _make_task(iso_db, 't_video', 'txt2video', {}, ['clip_score'])
    assert _db.upsert_aigc_audio_report(iso_db, 't_video') is True
    has_errors, note = _flag(iso_db, 't_video')
    assert has_errors == 1 and 'clip_score' in note


def test_fvd_present_not_flagged(iso_db):
    """txt2video configured with fvd and fvd produced -> NOT flagged."""
    _make_task(iso_db, 't_fvd', 'txt2video', {'fvd': 123.4}, ['fvd'])
    assert _db.upsert_aigc_audio_report(iso_db, 't_fvd') is True
    assert _flag(iso_db, 't_fvd') == (0, '')