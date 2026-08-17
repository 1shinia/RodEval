"""DashScope 视频生成适配器单元测试（纯 mock，不触发真实 API）。

覆盖: 工厂创建 / URL 解析（两种 api_base 填法）/ headers / 嵌套 payload /
异步检测 / 轮询 URL / 轮询状态机 / 视频字节提取 / 参数转换 helpers。
"""
from unittest import mock

from evalscope.backend.aigc_eval.models.adapters import (
    DashScopeVideoAdapter,
    _nearest_dashscope_duration,
    _wh_to_aspect_ratio,
    create_adapter,
)

BASE_CONFIG = {
    'provider': 'dashscope',
    'api_base': 'https://dashscope.aliyuncs.com/api/v1',
    'api_key': 'sk-test',
    'model_name_or_path': 'kling/kling-v3-omni-video-generation',
}

SYNTHESIS_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis'


def fake_resp(status_code=200, json_data=None, content=b'FAKE_MP4'):
    r = mock.MagicMock()
    r.status_code = status_code
    r.ok = status_code < 400
    r.json.return_value = json_data or {}
    r.content = content
    return r


def make_adapter(**overrides):
    return DashScopeVideoAdapter({**BASE_CONFIG, **overrides})


def test_factory_creates_dashscope_adapter():
    assert isinstance(create_adapter(BASE_CONFIG), DashScopeVideoAdapter)


def test_resolve_url_from_api_root():
    assert make_adapter()._resolve_url('txt2video') == SYNTHESIS_URL


def test_resolve_url_tolerates_full_endpoint():
    a = make_adapter(api_base=SYNTHESIS_URL)
    assert a._resolve_url('txt2video') == SYNTHESIS_URL


def test_build_headers():
    h = make_adapter()._build_headers()
    assert h['X-DashScope-Async'] == 'enable'
    assert h['Authorization'] == 'Bearer sk-test'


def test_build_payload_nested_structure():
    p = make_adapter()._build_payload('一只小猫在月光下奔跑', tool='txt2video',
                                      width=1280, height=720, num_frames=16, fps=8, ratio='16:9')
    assert p['model'] == 'kling/kling-v3-omni-video-generation'
    assert p['input'] == {'prompt': '一只小猫在月光下奔跑'}
    assert p['parameters']['aspect_ratio'] == '16:9'
    assert p['parameters']['duration'] == 5  # 16//8=2 -> snap to 5
    assert p['parameters']['mode'] == 'std'
    assert not any(k in p for k in ('prompt', 'size', 'n'))


def test_build_payload_wh_fallback_and_duration():
    a = make_adapter()
    assert a._build_payload('x', tool='txt2video', width=1080, height=1920, num_frames=240, fps=24)['parameters']['aspect_ratio'] == '9:16'
    assert a._build_payload('x', tool='txt2video', width=1080, height=1920, num_frames=240, fps=24)['parameters']['duration'] == 10
    assert a._build_payload('x', tool='txt2video', width=960, height=720)['parameters']['aspect_ratio'] == '4:3'


def test_build_payload_rejects_non_video_tool():
    import pytest
    with pytest.raises(ValueError):
        make_adapter()._build_payload('x', tool='txt2img')


def test_async_detection():
    a = make_adapter()
    assert a._is_async_response({'output': {'task_id': 't1', 'task_status': 'PENDING'}, 'request_id': 'r'})
    assert not a._is_async_response({'task_id': 't1', 'status': 'PENDING'})
    assert not a._is_async_response({})


def test_poll_url():
    assert make_adapter()._resolve_poll_url('http://post', 'task-123') == \
        'https://dashscope.aliyuncs.com/api/v1/tasks/task-123'


def test_poll_immediate_success():
    done = {'output': {'task_id': 't1', 'task_status': 'SUCCEEDED', 'video_url': 'https://cdn/v.mp4'}}
    assert make_adapter()._poll_async(done, 'http://post', {}) == done


def test_poll_state_machine():
    seq = [
        fake_resp(200, {'output': {'task_id': 't1', 'task_status': 'RUNNING'}}),
        fake_resp(200, {'output': {'task_id': 't1', 'task_status': 'SUCCEEDED', 'video_url': 'https://cdn/v.mp4'}}),
    ]
    with mock.patch('time.sleep'), mock.patch('requests.get', side_effect=seq) as mg:
        result = make_adapter()._poll_async(
            {'output': {'task_id': 't1', 'task_status': 'PENDING'}}, 'http://post', {})
    assert result['output']['task_status'] == 'SUCCEEDED'
    assert mg.call_args[0][0] == 'https://dashscope.aliyuncs.com/api/v1/tasks/t1'


def test_poll_failure_raises():
    import pytest
    with mock.patch('time.sleep'), mock.patch('requests.get',
            return_value=fake_resp(200, {'output': {'task_id': 't1', 'task_status': 'FAILED', 'message': 'model not found'}})):
        with pytest.raises(RuntimeError, match='model not found'):
            make_adapter()._poll_async(
                {'output': {'task_id': 't1', 'task_status': 'RUNNING'}}, 'http://post', {})


def test_extract_bytes_downloads_video():
    with mock.patch('requests.get', return_value=fake_resp(200, {}, content=b'REAL_MP4')) as mg:
        data = {'output': {'task_id': 't1', 'task_status': 'SUCCEEDED', 'video_url': 'https://cdn/v.mp4'}}
        assert make_adapter()._extract_bytes(data) == b'REAL_MP4'
        assert mg.call_args[0][0] == 'https://cdn/v.mp4'


def test_extract_bytes_missing_url_raises():
    import pytest
    with pytest.raises(ValueError, match='Cannot extract'):
        make_adapter()._extract_bytes({'output': {'task_id': 't1', 'task_status': 'PENDING'}})


def test_aspect_ratio_and_duration_helpers():
    assert _wh_to_aspect_ratio(1280, 720) == '16:9'
    assert _wh_to_aspect_ratio(854, 480) == '16:9'
    assert _wh_to_aspect_ratio(960, 720) == '4:3'
    assert _nearest_dashscope_duration(2) == 5
    assert _nearest_dashscope_duration(5) == 5
    assert _nearest_dashscope_duration(10) == 10
    assert _nearest_dashscope_duration(24) == 10
