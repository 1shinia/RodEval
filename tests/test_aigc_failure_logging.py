"""AIGC 任务失败原因必须落到任务日志。

页面只通过 SSE 展示任务日志（outputs/<task>/logs/*.log），progress.json 里只有
一句 "Task failed"。历史上适配器抛出的真实原因（例如百炼 403 的 message/request_id）
只出现在服务端日志，用户自查时看不到。
"""
import logging
import pytest
from unittest import mock

from evalscope.backend.aigc_eval import backend_manager as bm


def _manager(config):
    mgr = bm.AIGCBackendManager.__new__(bm.AIGCBackendManager)  # 跳过 __init__（不需要真实 config 文件）
    mgr._parse_config = lambda: config  # type: ignore[method-assign]
    return mgr


def test_run_logs_failure_reason_for_txt2video(caplog):
    config = mock.MagicMock(
        tool='txt2video',
        model=mock.MagicMock(model_name_or_path='wan2.7-t2v'),
        eval=mock.MagicMock(metrics=['clip_score']),
    )
    mgr = _manager(config)
    mgr._run_txt2video = mock.MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError('HTTP 403 AccessDenied current user api does not support asynchronous calls')
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match='does not support asynchronous calls'):
        mgr.run()

    # 原因写进任务日志（同一 logger → 任务日志文件），且异常继续向上抛（父进程仍能标记失败）
    assert 'AIGC txt2video 任务失败' in caplog.text
    assert 'does not support asynchronous calls' in caplog.text


def test_run_propagates_without_swallowing():
    config = mock.MagicMock(
        tool='txt2img',
        model=mock.MagicMock(model_name_or_path='sd-v1-5'),
        eval=mock.MagicMock(metrics=['clip_score']),
    )
    mgr = _manager(config)
    mgr._run_txt2img = mock.MagicMock(side_effect=ValueError('bad payload'))  # type: ignore[method-assign]

    with pytest.raises(ValueError, match='bad payload'):
        mgr.run()
