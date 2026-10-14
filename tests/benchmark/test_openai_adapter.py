"""OpenAI 兼容视频来源适配器单元测试（纯 mock，不触发真实 API）。

覆盖: 文生视频 payload 形态（metadata 契约）/ 时长显式校验 /
异步识别（含 {code,message,data} 包裹形态）/ 轮询预算 / 状态词归一（大小写 + 同义终态词）/
非 2xx 脱敏回显 / 401-403 快速失败 / 404 宽限后报「查询地址配错」/ 超时留痕 /
创建响应即终态时仍取内容模板 / 轮询终态 result_url 取流 / 成品下载 401-403 带鉴权回退。

同时守护一条边界：DashScope 与 Volcengine 两个来源各自持有原生轮询实现，
基类默认轮询的改动只落在 OpenAI 兼容来源上。
"""
import base64
import json
import logging
from unittest import mock

import pytest

from evalscope.backend.aigc_eval.models.adapters import (
    AIGCApiAdapter,
    DashScopeVideoAdapter,
    OpenAICompatibleAdapter,
    VolcengineVideoAdapter,
)

BASE_CONFIG = {
    'provider': 'openai',
    'api_base': 'https://api.example.test/v1',
    'api_key': 'sk-test-key',
    'model_name_or_path': 'video-model',
    'model_type': 'txt2video',
}

POST_URL = 'https://api.example.test/v1/video/generations'


def fake_resp(status_code=200, json_data=None, content=b'FAKE_MP4', text=''):
    r = mock.MagicMock()
    r.status_code = status_code
    r.ok = status_code < 400
    r.json.return_value = json_data or {}
    r.content = content
    r.text = text
    return r


def make_adapter(**overrides):
    return OpenAICompatibleAdapter({**BASE_CONFIG, **overrides})


def route_responses(routes):
    """按 URL 分派 GET 响应；未登记的 URL 直接断言失败，避免测试静默走错分支。"""

    def _side_effect(url, *args, **kwargs):
        if url not in routes:
            raise AssertionError(f'unexpected GET {url}')
        return routes[url]

    return _side_effect


# ── 文生视频 payload 形态：分辨率/画幅比/时长在 metadata 内 ─────────────

VIDEO_METADATA_DEFAULTS = {
    'content': [],
    'resolution': '720p',  # 由 1280x720 推导
    'ratio': '16:9',
    'duration': 5,
    'generate_audio': False,
}


def test_video_payload_nests_gateway_contract_metadata():
    """网关契约（UniToken/new-api 系）：metadata 是唯一被解析的载荷，
    顶层 size/n/resolution 网关都不读（实测 400 "resolution is required"）。"""
    payload = make_adapter()._build_payload('一只猫在跑', tool='txt2video', width=1280, height=720)
    assert set(payload) == {'model', 'prompt', 'metadata'}
    assert payload['metadata'] == VIDEO_METADATA_DEFAULTS


def test_video_payload_keeps_explicit_resolution_ratio_and_audio():
    payload = make_adapter()._build_payload(
        '一只猫在跑',
        tool='txt2video',
        width=1280,
        height=720,
        resolution='1080p',
        ratio='9:16',
        duration=12,
        generate_audio=True,
    )
    meta = payload['metadata']
    assert (meta['resolution'], meta['ratio'], meta['duration']) == ('1080p', '9:16', 12)
    assert meta['generate_audio'] is True


def test_video_payload_carries_seed_only_when_given():
    """seed 落在 metadata（与分辨率同层）：图片侧走顶层，视频按网关契约只能进 metadata。

    网关支持就生效；不支持多半随 metadata 被忽略，对方若明确以 400 点名 seed，
    由 _handle_retry 去掉后重发一次 —— 不因这个可选字段拖垮出片。
    """
    adapter = make_adapter()
    given = adapter._build_payload('一只猫在跑', tool='txt2video', width=1280, height=720, seed=1234)
    assert given['metadata']['seed'] == 1234
    absent = adapter._build_payload('一只猫在跑', tool='txt2video', width=1280, height=720)
    assert 'seed' not in absent['metadata']


def test_video_duration_defaults_to_five():
    payload = make_adapter()._build_payload('一只猫在跑', tool='txt2video', width=1280, height=720)
    assert payload['metadata']['duration'] == 5


def test_video_duration_accepts_range_bounds_and_numeric_string():
    adapter = make_adapter()
    for given, expected in ((5, 5), (15, 15), (10, 10), ('10', 10), ('5', 5)):
        payload = adapter._build_payload('一只猫在跑', tool='txt2video', duration=given)
        assert payload['metadata']['duration'] == expected


@pytest.mark.parametrize('given', [3, 16, 0, 100, -5, 'abc', '', 7.5, None])
def test_video_duration_rejects_invalid_values(given):
    """空值 None/'' 走默认 5；其余非法值报错而不是被悄悄改成 5。"""
    adapter = make_adapter()
    if given is None or given == '':
        assert adapter._build_payload('x', tool='txt2video', duration=given)['metadata']['duration'] == 5
        return
    with pytest.raises(ValueError, match='5~15'):
        adapter._build_payload('x', tool='txt2video', duration=given)


def test_video_duration_rejects_bool():
    with pytest.raises(ValueError, match='5~15'):
        make_adapter()._build_payload('x', tool='txt2video', duration=True)


# ── 400 兜底：只做窄口处理，其余原样上抛 ─────────────────────────────


def test_video_seed_rejected_by_gateway_is_retried_without_seed():
    adapter = make_adapter()
    payload = adapter._build_payload('x', tool='txt2video', width=1280, height=720, seed=1234)
    rejected = fake_resp(400, text='{"code":"invalid_request","message":"seed is not supported"}')
    with mock.patch('requests.post', return_value=fake_resp(200, {'id': 'v-1', 'status': 'queued'})) as post:
        out = adapter._handle_retry(rejected, POST_URL, {}, payload, tool='txt2video')
    assert out.status_code == 200
    sent = post.call_args.kwargs['json']
    assert 'seed' not in sent['metadata'], '重发必须去掉 seed'
    assert sent['metadata']['resolution'] == '720p'
    assert payload['metadata']['seed'] == 1234, '不得就地修改原 payload'


def test_video_400_not_mentioning_seed_is_surfaced_untouched():
    """其它 400 不重试：分辨率缺失这类必须原样回显，别用改写请求掩盖真实原因。"""
    adapter = make_adapter()
    payload = adapter._build_payload('x', tool='txt2video', width=1280, height=720, seed=1234)
    rejected = fake_resp(400, text='{"code":"invalid_request","message":"resolution is required"}')
    with mock.patch('requests.post') as post:
        out = adapter._handle_retry(rejected, POST_URL, {}, payload, tool='txt2video')
    assert out is rejected
    assert post.call_count == 0


def test_image_size_fallback_survives_the_video_guard():
    adapter = make_adapter()
    payload = adapter._build_payload('x', tool='txt2img', width=1280, height=720)
    rejected = fake_resp(400, text='{"error":{"message":"size is not supported"}}')
    with mock.patch('requests.post', return_value=fake_resp(200)) as post:
        adapter._handle_retry(rejected, POST_URL, {}, payload, tool='txt2img', width=1280, height=720)
    assert post.call_args.kwargs['json']['size'] == '720p'


# ── 轮询预算 ─────────────────────────────────────────────────────────


def test_poll_budget_covers_real_generation_time():
    adapter = make_adapter()
    attempts = adapter.POLL_TIMEOUT_SECONDS // adapter.POLL_INTERVAL_SECONDS
    assert adapter.POLL_INTERVAL_SECONDS == 15
    assert attempts == 180, '旧的 60×5s=300s 会把仍在生成的任务误判为超时'
    assert adapter.POLL_TIMEOUT_SECONDS >= 45 * 60


# ── 异步识别 ─────────────────────────────────────────────────────────


def test_async_response_detected_for_flat_and_wrapped_shapes():
    adapter = make_adapter()
    assert adapter._is_async_response({'id': 'v-1', 'status': 'queued'}) is True
    assert adapter._is_async_response({'task_id': 't-1', 'status': 'queued'}) is True
    assert adapter._is_async_response({'code': 0, 'message': 'ok',
                                      'data': {'task_id': 't-1', 'status': 'queued'}}) is True
    assert adapter._is_async_response({'data': [{'url': 'http://cdn/v.mp4'}]}) is False


@pytest.mark.parametrize('status', ['SUCCEEDED', 'Success', 'FINISHED', 'completed', ' done '])
def test_immediate_terminal_status_skips_polling(status):
    """终态词大小写/同义写法都要认，否则会空转到超时。"""
    adapter = make_adapter()
    with mock.patch('requests.get') as mg:
        result = adapter._poll_async({'id': 'v-1', 'status': status}, POST_URL, {})
    assert result['status'] == status
    mg.assert_not_called()


def test_poll_state_machine_extracts_final_media():
    adapter = make_adapter()
    poll_url = 'https://api.example.test/v1/video/generations/v-1'
    media_url = 'http://cdn.example/v.mp4'
    poll_sequence = [
        fake_resp(200, {'id': 'v-1', 'status': 'queued'}),
        fake_resp(200, {'id': 'v-1', 'status': 'IN_PROGRESS', 'progress': 42}),
        fake_resp(200, {'id': 'v-1', 'status': 'completed', 'url': media_url}),
    ]
    calls = []

    def _side_effect(url, *args, **kwargs):
        calls.append(url)
        if url == poll_url:
            return poll_sequence.pop(0)
        if url == media_url:
            return fake_resp(200, content=b'FAKE_MP4')
        raise AssertionError(f'unexpected GET {url}')

    with mock.patch('time.sleep'), mock.patch('requests.get', side_effect=_side_effect):
        result = adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
        assert adapter._extract_bytes(result) == b'FAKE_MP4'

    assert calls == [poll_url, poll_url, poll_url, media_url]


def test_poll_extracts_result_url_from_wrapped_terminal_response():
    """网关轮询终态形态：{code,message,data:{status:SUCCESS, result_url}}。"""
    adapter = make_adapter()
    media_url = 'http://cdn.example/v.mp4'
    poll_body = {'code': 'success', 'message': '', 'data': {
        'id': 'v-1', 'task_id': 'v-1', 'status': 'SUCCESS', 'progress': '100%', 'result_url': media_url}}
    responses = [fake_resp(200, poll_body), fake_resp(200, content=b'FAKE_MP4')]
    with mock.patch('time.sleep'), mock.patch('requests.get', side_effect=responses) as mg:
        result = adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
        assert adapter._extract_bytes(result) == b'FAKE_MP4'
    assert mg.call_args_list[0][0][0] == 'https://api.example.test/v1/video/generations/v-1'
    assert mg.call_args_list[-1][0][0] == media_url


def test_poll_unwraps_wrapped_create_response_for_task_id():
    adapter = make_adapter()
    url = 'https://api.example.test/v1/video/generations/t-9'
    wrapped = {'code': 0, 'message': 'ok', 'data': {'task_id': 't-9', 'status': 'queued'}}
    responses = [fake_resp(200, {'status': 'success', 'url': 'http://cdn.example/v.mp4'}),
                 fake_resp(200, content=b'FAKE_MP4')]
    with mock.patch('time.sleep'), mock.patch('requests.get', side_effect=responses) as mg:
        result = adapter._poll_async(wrapped, POST_URL, {})
        assert adapter._extract_bytes(result) == b'FAKE_MP4'
    assert mg.call_args_list[0][0][0] == url


# ── 非 2xx：脱敏回显 / 快速失败 / 404 宽限 ────────────────────────────


def test_poll_non_2xx_logs_redacted_body_and_keeps_polling(caplog):
    adapter = make_adapter()
    body = json.dumps({'code': 'InternalError', 'message': 'boom', 'trace': 'sk-abcdef123456'})
    responses = [fake_resp(500, text=body),
                 fake_resp(200, {'id': 'v-1', 'status': 'success', 'url': 'http://cdn.example/v.mp4'}),
                 fake_resp(200, content=b'FAKE_MP4')]
    with caplog.at_level(logging.WARNING), mock.patch('time.sleep'), \
            mock.patch('requests.get', side_effect=responses):
        result = adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
        assert adapter._extract_bytes(result) == b'FAKE_MP4'

    assert 'boom' in caplog.text
    assert 'sk-abcdef123456' not in caplog.text, '响应体里的凭据摘要必须脱敏'
    assert '***' in caplog.text


def test_poll_auth_failure_fails_fast():
    adapter = make_adapter()
    body = json.dumps({'code': 'InvalidApiKey', 'message': 'invalid api key'})
    with mock.patch('time.sleep'), \
            mock.patch('requests.get', return_value=fake_resp(401, text=body)) as mg:
        with pytest.raises(RuntimeError) as excinfo:
            adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
    assert '不可重试' in str(excinfo.value)
    assert 'InvalidApiKey' in str(excinfo.value)
    assert mg.call_count == 1, '鉴权失败不会自愈，不该重试到超时'


def test_poll_not_found_grace_then_actionable_error():
    adapter = make_adapter()
    with mock.patch('time.sleep'), \
            mock.patch('requests.get', return_value=fake_resp(404, {})) as mg:
        with pytest.raises(RuntimeError) as excinfo:
            adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
    message = str(excinfo.value)
    assert 'async_poll_url' in message
    assert '404' in message
    assert mg.call_count == adapter.POLL_NOT_FOUND_GRACE


def test_poll_timeout_reports_last_status_and_recheck_url():
    adapter = make_adapter()
    running = fake_resp(200, {'id': 'v-1', 'status': 'RUNNING'})
    with mock.patch.object(OpenAICompatibleAdapter, 'POLL_TIMEOUT_SECONDS', 30), \
            mock.patch.object(OpenAICompatibleAdapter, 'POLL_INTERVAL_SECONDS', 15), \
            mock.patch('time.sleep'), mock.patch('requests.get', return_value=running):
        with pytest.raises(TimeoutError) as excinfo:
            adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
    message = str(excinfo.value)
    assert 'last status=running' in message
    assert 'GET https://api.example.test/v1/video/generations/v-1' in message
    assert '请勿重复创建任务' in message


def test_poll_failure_reports_code_and_message_redacted():
    adapter = make_adapter()
    failed = fake_resp(200, {
        'id': 'v-1',
        'status': 'FAILED',
        'error': {'code': 'InvalidParameter', 'message': 'ratio not supported (trace sk-abcdef123456)'},
    })
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=failed):
        with pytest.raises(RuntimeError) as excinfo:
            adapter._poll_async({'id': 'v-1', 'status': 'queued'}, POST_URL, {})
    message = str(excinfo.value)
    assert 'InvalidParameter' in message
    assert 'ratio not supported' in message
    assert 'sk-abcdef123456' not in message


# ── 创建即终态：内容模板仍要生效 ─────────────────────────────────────


def test_immediate_terminal_status_still_fetches_content_template():
    adapter = make_adapter(async_content_url='video/generations/{id}/content')
    content_url = 'https://api.example.test/v1/video/generations/v-7/content'
    with mock.patch('requests.get', side_effect=route_responses(
            {content_url: fake_resp(200, content=b'MP4BYTES')})) as mg:
        result = adapter._poll_async({'id': 'v-7', 'status': 'completed'}, POST_URL, {})
        assert base64.b64decode(result['b64_json']) == b'MP4BYTES'
    assert mg.call_count == 1
    assert mg.call_args_list[0][0][0] == content_url


# ── 成品下载：401/403 带鉴权回退 ─────────────────────────────────────


def test_media_download_retries_with_bearer_on_401():
    adapter = make_adapter()
    responses = [fake_resp(401, text='unauthorized'), fake_resp(200, content=b'AUTHED_MP4')]
    with mock.patch('requests.get', side_effect=responses) as mg:
        data = {'data': [{'url': 'https://cdn.example/v.mp4'}]}
        assert adapter._extract_bytes(data) == b'AUTHED_MP4'
    assert mg.call_count == 2
    assert mg.call_args_list[1][1]['headers'] == {'Authorization': 'Bearer sk-test-key'}


def test_media_download_does_not_retry_401_without_api_key():
    adapter = make_adapter(api_key='')
    resp = fake_resp(401, text='unauthorized')
    with mock.patch('requests.get', return_value=resp) as mg:
        adapter._extract_bytes({'data': [{'url': 'https://cdn.example/v.mp4'}]})
    assert mg.call_count == 1
    resp.raise_for_status.assert_called_once()


def test_media_download_public_url_is_single_get():
    adapter = make_adapter()
    with mock.patch('requests.get', return_value=fake_resp(200, content=b'OPEN_MP4')) as mg:
        assert adapter._extract_bytes({'data': [{'url': 'https://cdn.example/v.mp4'}]}) == b'OPEN_MP4'
    assert mg.call_count == 1


# ── 边界：另外两个来源仍走各自的轮询实现 ─────────────────────────────


def test_frozen_providers_keep_their_own_polling():
    assert DashScopeVideoAdapter._poll_async is not AIGCApiAdapter._poll_async
    assert VolcengineVideoAdapter._poll_async is not AIGCApiAdapter._poll_async
    assert DashScopeVideoAdapter.SUCCESS_STATUSES == ('SUCCEEDED', )
    assert VolcengineVideoAdapter.SUCCESS_STATUSES == ('succeeded', 'success', 'completed', 'done')
    assert DashScopeVideoAdapter.POLL_TIMEOUT_SECONDS == 2700
    assert VolcengineVideoAdapter.POLL_TIMEOUT_SECONDS == 2700
