"""DashScope 视频生成适配器单元测试（纯 mock，不触发真实 API）。

覆盖: 工厂创建 / URL 解析（两种 api_base 填法）/ headers / 嵌套 payload /
异步检测 / 轮询 URL / 轮询状态机 / 视频字节提取 / 参数转换 helpers。
"""
from unittest import mock

from evalscope.backend.aigc_eval.models.adapters import (
    DashScopeVideoAdapter,
    OpenAICompatibleAdapter,
    VolcengineVideoAdapter,
    _nearest_dashscope_duration,
    _wh_to_aspect_ratio,
    create_adapter,
)

VOLC_CONFIG = {
    'provider': 'volcengine',
    'api_base': 'https://ark.cn-beijing.volces.com/api/v3',
    'api_key': 'sk-test',
    'model_name_or_path': 'doubao-seedance-2-0-260128',
    'model_type': 'txt2video',
}


def make_volc_adapter(**overrides):
    return VolcengineVideoAdapter({**VOLC_CONFIG, **overrides})


def test_volcengine_video_payload_preserves_valid_duration():
    assert make_volc_adapter()._build_payload('x', tool='txt2video', duration=5)['duration'] == 5
    assert make_volc_adapter()._build_payload('x', tool='txt2video', duration=15)['duration'] == 15


def test_volcengine_video_payload_rejects_duration_outside_model_range():
    import pytest
    with pytest.raises(ValueError, match='5～15'):
        make_volc_adapter()._build_payload('x', tool='txt2video', duration=4)
    with pytest.raises(ValueError, match='5～15'):
        make_volc_adapter()._build_payload('x', tool='txt2video', duration=16)


def test_factory_creates_volcengine_video_adapter():
    assert isinstance(create_adapter(VOLC_CONFIG), VolcengineVideoAdapter)


def test_volcengine_video_payload_uses_native_contract():
    payload = make_volc_adapter()._build_payload(
        '一只老虎在森林里奔跑', tool='txt2video', width=1280, height=720,
        duration=10, seed=42,
    )
    assert payload['model'] == 'doubao-seedance-2-0-260128'
    assert payload['content'] == [{'type': 'text', 'text': '一只老虎在森林里奔跑'}]
    assert payload['resolution'] == '720p'
    assert payload['ratio'] == '16:9'
    assert payload['duration'] == 10
    assert payload['seed'] == 42
    assert 'prompt' not in payload
    assert 'size' not in payload
    assert 'n' not in payload


def test_volcengine_video_urls_and_extract():
    adapter = make_volc_adapter()
    assert adapter._resolve_url('txt2video').endswith('/contents/generations/tasks')
    done = {'id': 'cgt-1', 'status': 'succeeded', 'content': {'video_url': 'https://cdn/v.mp4'}}
    with mock.patch('requests.get', return_value=mock.MagicMock(
        content=b'VOLC_MP4', ok=True, status_code=200,
    )) as mocked_get:
        assert adapter._extract_bytes(done) == b'VOLC_MP4'
        assert mocked_get.call_args.args[0] == 'https://cdn/v.mp4'


def test_openai_video_payload_uses_legacy_size_and_explicit_duration():
    adapter = OpenAICompatibleAdapter({
        'api_base': 'https://api.example.test/v1',
        'api_key': 'sk-test',
        'model_name_or_path': 'video-model',
    })
    payload = adapter._build_payload(
        '森林里一只老虎在奔跑',
        tool='txt2video',
        width=1280,
        height=720,
        duration=10,
    )
    assert payload['n'] == 1
    assert payload['size'] == '1280x720'
    assert payload['duration'] == 10
    assert 'resolution' not in payload
    assert 'ratio' not in payload


def test_openai_video_payload_preserves_explicit_resolution_and_ratio():
    adapter = OpenAICompatibleAdapter({
        'api_base': 'https://api.example.test/v1',
        'api_key': 'sk-test',
        'model_name_or_path': 'video-model',
    })
    payload = adapter._build_payload(
        '森林里一只老虎在奔跑',
        tool='txt2video',
        width=1280,
        height=720,
        duration=10,
        resolution='720p',
        ratio='16:9',
    )
    assert payload['resolution'] == '720p'
    assert payload['ratio'] == '16:9'
    assert payload['duration'] == 10


def test_openai_image_payload_forwards_generation_parameters():
    adapter = OpenAICompatibleAdapter({
        'api_base': 'https://api.example.test/v1',
        'api_key': 'sk-test',
        'model_name_or_path': 'image-model',
    })
    payload = adapter._build_payload(
        '一只小狗',
        tool='txt2img',
        width=1024,
        height=1024,
        num_inference_steps=8,
        guidance_scale=7.5,
        seed=43,
    )
    assert payload['num_inference_steps'] == 8
    assert payload['guidance_scale'] == 7.5
    assert payload['seed'] == 43


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
                                      width=1280, height=720, duration=10, ratio='16:9')
    assert p['model'] == 'kling/kling-v3-omni-video-generation'
    assert p['input'] == {'prompt': '一只小猫在月光下奔跑'}
    assert p['parameters']['aspect_ratio'] == '16:9'
    assert p['parameters']['duration'] == 10
    assert p['parameters']['mode'] == 'std'
    assert not any(k in p for k in ('prompt', 'size', 'n'))


def test_build_payload_wh_fallback_and_duration():
    a = make_adapter()
    assert a._build_payload('x', tool='txt2video', width=1080, height=1920, duration=10)['parameters']['aspect_ratio'] == '9:16'
    assert a._build_payload('x', tool='txt2video', width=1080, height=1920, duration=10)['parameters']['duration'] == 10
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
