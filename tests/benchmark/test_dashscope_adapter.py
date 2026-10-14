"""DashScope 视频生成适配器单元测试（纯 mock，不触发真实 API）。

覆盖: 工厂创建 / URL 解析（两种 api_base 填法）/ headers / 嵌套 payload /
异步检测 / 轮询 URL / 轮询状态机 / 视频字节提取 / 参数转换 helpers。
"""
import pytest
from unittest import mock

from evalscope.backend.aigc_eval.models.adapters import (
    _DASHSCOPE_LEGACY_SIZE_TABLE,
    _DASHSCOPE_VIDEO_FAMILIES,
    DashScopeVideoAdapter,
    OpenAICompatibleAdapter,
    VolcengineVideoAdapter,
    _dashscope_duration,
    _dashscope_ratios,
    _dashscope_video_family,
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
    # 官方：Seedance 2.0 系列时长 4~15 秒（doubao-seedance-2-0-260128 见 VOLC_CONFIG）
    assert make_volc_adapter()._build_payload('x', tool='txt2video', duration=4)['duration'] == 4
    assert make_volc_adapter()._build_payload('x', tool='txt2video', duration=15)['duration'] == 15


def test_volcengine_video_payload_rejects_duration_outside_model_range():
    with pytest.raises(ValueError, match='4～15'):
        make_volc_adapter()._build_payload('x', tool='txt2video', duration=3)
    with pytest.raises(ValueError, match='4～15'):
        make_volc_adapter()._build_payload('x', tool='txt2video', duration=16)


def test_volcengine_duration_range_is_per_model():
    """官方模型列表：2.5 = 4~30；2.0 系列 = 4~15；1.0 pro / pro fast = 2~12。"""
    five = make_volc_adapter(model_name_or_path='doubao-seedance-2-5-260628')
    assert five._build_payload('x', tool='txt2video', duration=30)['duration'] == 30
    with pytest.raises(ValueError, match='4～30'):
        five._build_payload('x', tool='txt2video', duration=31)

    one = make_volc_adapter(model_name_or_path='doubao-seedance-1-0-pro-fast-251015')
    assert one._build_payload('x', tool='txt2video', duration=2)['duration'] == 2
    with pytest.raises(ValueError, match='2～12'):
        one._build_payload('x', tool='txt2video', duration=13)


def test_volcengine_smart_duration_only_for_2x_series():
    """官方：duration = -1（模型智能选择）仅 2.0 / 2.5 系列支持。"""
    assert make_volc_adapter()._build_payload('x', tool='txt2video', duration=-1)['duration'] == -1
    with pytest.raises(ValueError, match='2～12'):
        make_volc_adapter(model_name_or_path='doubao-seedance-1-0-pro-250528')._build_payload(
            'x', tool='txt2video', duration=-1
        )


def test_volcengine_resolution_is_per_model():
    """官方模型列表：2.0-fast / 2.0-mini 仅 480p、720p；2.0 支持 4k；2.5 无 4k。"""
    fast = make_volc_adapter(model_name_or_path='doubao-seedance-2-0-fast-260128')
    assert fast._build_payload('x', tool='txt2video', resolution='480p')['resolution'] == '480p'
    with pytest.raises(ValueError, match='分辨率'):
        fast._build_payload('x', tool='txt2video', resolution='1080p')

    assert make_volc_adapter()._build_payload('x', tool='txt2video', resolution='4k')['resolution'] == '4k'
    five = make_volc_adapter(model_name_or_path='doubao-seedance-2-5-260628')
    with pytest.raises(ValueError, match='分辨率'):
        five._build_payload('x', tool='txt2video', resolution='4k')


def test_volcengine_unknown_model_rejected():
    """无法识别的火山视频模型直接报错，不猜参数形态。"""
    with pytest.raises(ValueError, match='无法识别火山引擎视频模型'):
        make_volc_adapter(model_name_or_path='doubao-seedance-9-9-xxxx')._build_payload('x', tool='txt2video')


def test_volcengine_ratio_defaults_to_16_9():
    """官方：文生视频默认 16:9；省略 ratio 时由后端补默认值。"""
    assert make_volc_adapter()._build_payload('x', tool='txt2video')['ratio'] == '16:9'
    assert make_volc_adapter()._build_payload('x', tool='txt2video', ratio='21:9')['ratio'] == '21:9'


def test_volcengine_ratio_is_validated_per_model():
    """官方可选值：16:9、4:3、1:1、3:4、9:16、21:9（+adaptive，仅 2.5/2.0 系列）。"""
    with pytest.raises(ValueError, match='不支持画幅比 5:4'):
        make_volc_adapter()._build_payload('x', tool='txt2video', ratio='5:4')
    # adaptive：2.0 / 2.5 放行，1.0 系列文生视频不支持
    assert make_volc_adapter()._build_payload('x', tool='txt2video', ratio='adaptive')['ratio'] == 'adaptive'
    five = make_volc_adapter(model_name_or_path='doubao-seedance-2-5-260628')
    assert five._build_payload('x', tool='txt2video', ratio='adaptive')['ratio'] == 'adaptive'
    one = make_volc_adapter(model_name_or_path='doubao-seedance-1-0-pro-250528')
    with pytest.raises(ValueError, match='不支持画幅比 adaptive'):
        one._build_payload('x', tool='txt2video', ratio='adaptive')
    assert one._build_payload('x', tool='txt2video', ratio='16:9')['ratio'] == '16:9'


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


def test_volcengine_poll_budget_covers_real_generation_time():
    """旧的 60×5s=300s 硬上限对视频生成太短：个人账号非 4k 并发仅 3，1080p / 长时长排队
    会让成功任务被误判为失败。现与百炼同源取 45 分钟。"""
    assert VolcengineVideoAdapter.POLL_INTERVAL_SECONDS == 15
    assert VolcengineVideoAdapter.POLL_TIMEOUT_SECONDS >= 2700
    assert VolcengineVideoAdapter.POLL_TIMEOUT_SECONDS // VolcengineVideoAdapter.POLL_INTERVAL_SECONDS >= 120


def test_volcengine_poll_immediate_success_skips_http():
    done = {'id': 'cgt-1', 'status': 'succeeded', 'content': {'video_url': 'https://cdn/v.mp4'}}
    with mock.patch('requests.get') as mg:
        assert make_volc_adapter()._poll_async(done, 'http://post', {}) == done
    assert mg.call_count == 0


def test_volcengine_poll_state_machine():
    seq = [
        fake_resp(200, {'id': 'cgt-1', 'status': 'running'}),
        fake_resp(200, {'id': 'cgt-1', 'status': 'succeeded', 'content': {'video_url': 'https://cdn/v.mp4'}}),
    ]
    with mock.patch('time.sleep'), mock.patch('requests.get', side_effect=seq) as mg:
        result = make_volc_adapter()._poll_async({'id': 'cgt-1', 'status': 'queued'}, 'http://post', {})
    assert result['status'] == 'succeeded'
    assert mg.call_args[0][0].endswith('/contents/generations/tasks/cgt-1')


def test_volcengine_poll_failure_raises_with_provider_error():
    failed = fake_resp(200, {
        'id': 'cgt-1', 'status': 'failed',
        'error': {'code': 'InvalidParameter', 'message': 'ratio not supported'},
    })
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=failed):
        with pytest.raises(RuntimeError, match='ratio not supported'):
            make_volc_adapter()._poll_async({'id': 'cgt-1', 'status': 'queued'}, 'http://post', {})


def test_volcengine_poll_non_2xx_logs_redacted_body(caplog):
    """可重试的非 2xx 必须回显响应体（脱敏后），且单次抖动不终止任务。"""
    import logging
    body = '{"error":{"code":"InternalError","message":"upstream busy"},"api_key":"sk-live-abcdef123456"}'
    seq = [fake_resp(500, {}, text=body), fake_resp(200, {'id': 'cgt-1', 'status': 'succeeded'})]
    with caplog.at_level(logging.WARNING), mock.patch('time.sleep'), mock.patch('requests.get', side_effect=seq):
        result = make_volc_adapter()._poll_async({'id': 'cgt-1', 'status': 'queued'}, 'http://post', {})
    assert result['status'] == 'succeeded'
    log = caplog.text
    assert 'HTTP 500' in log and 'upstream busy' in log
    assert 'sk-live-abcdef123456' not in log and '***' in log


def test_volcengine_poll_auth_error_fails_fast():
    """创建任务已用同一 API-KEY 成功，轮询遇 401/403 不会自愈：立即失败，
    不能每 15s 重试到 45 分钟超时 —— 错误里要带 message 与 request_id。"""
    body = ('{"error":{"code":"AuthenticationError","message":"the API key is invalid"},'
            '"request_id":"req-abc-123"}')
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=fake_resp(403, {}, text=body)) as mg:
        with pytest.raises(RuntimeError) as ei:
            make_volc_adapter()._poll_async({'id': 'cgt-1', 'status': 'queued'}, 'http://post', {})
    assert mg.call_count == 1  # 第一次就失败，没有空转
    msg = str(ei.value)
    assert 'the API key is invalid' in msg and 'req-abc-123' in msg and 'cgt-1' in msg


def test_volcengine_poll_timeout_reports_last_status_and_recheck_url():
    """超时必须留可复查线索：task_id + 最后状态 + 查询 URL，且不重复创建任务。"""
    running = fake_resp(200, {'id': 'cgt-1', 'status': 'running'})
    adapter = make_volc_adapter()
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=running), \
            mock.patch.object(VolcengineVideoAdapter, 'POLL_TIMEOUT_SECONDS', 30), \
            mock.patch.object(VolcengineVideoAdapter, 'POLL_INTERVAL_SECONDS', 15):
        with pytest.raises(TimeoutError) as ei:
            adapter._poll_async({'id': 'cgt-1', 'status': 'queued'}, 'http://post', {})
    msg = str(ei.value)
    assert 'cgt-1' in msg and 'last status=running' in msg
    assert '/contents/generations/tasks/cgt-1' in msg


def test_openai_video_payload_nests_metadata_and_derives_defaults():
    """网关契约：metadata 唯一被解析，顶层 size/n/resolution 网关都不读
    （实测该形态返回 400 "resolution is required"）。"""
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
    assert set(payload) == {'model', 'prompt', 'metadata'}
    assert payload['metadata'] == {
        'content': [],
        'resolution': '720p',
        'ratio': '16:9',
        'duration': 10,
        'generate_audio': False,
    }


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
    assert payload['metadata']['resolution'] == '720p'
    assert payload['metadata']['ratio'] == '16:9'
    assert payload['metadata']['duration'] == 10
    assert 'resolution' not in payload and 'size' not in payload and 'n' not in payload


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


def fake_resp(status_code=200, json_data=None, content=b'FAKE_MP4', text=''):
    r = mock.MagicMock()
    r.status_code = status_code
    r.ok = status_code < 400
    r.json.return_value = json_data or {}
    r.content = content
    r.text = text
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


WAN27_CONFIG = {**BASE_CONFIG, 'model_name_or_path': 'wan2.7-t2v'}
WAN26_CONFIG = {**BASE_CONFIG, 'model_name_or_path': 'wan2.6-t2v'}


def make_wan27(**overrides):
    return DashScopeVideoAdapter({**WAN27_CONFIG, **overrides})


def make_wan26(**overrides):
    return DashScopeVideoAdapter({**WAN26_CONFIG, **overrides})


# ── kling 分支 ──────────────────────────────────────────────────────────

def test_kling_payload_uses_mode_and_defaults_to_pro():
    p = make_adapter()._build_payload('一只小猫在月光下奔跑', tool='txt2video',
                                      width=1280, height=720, duration=10, ratio='16:9')
    assert p['model'] == 'kling/kling-v3-omni-video-generation'
    assert p['input'] == {'prompt': '一只小猫在月光下奔跑'}
    assert p['parameters']['aspect_ratio'] == '16:9'
    assert p['parameters']['duration'] == 10
    # 默认档位 1080p → 官方 mode=pro（旧实现写死 std 即 720P）
    assert p['parameters']['mode'] == 'pro'
    assert p['parameters']['watermark'] is False
    assert 'seed' not in p['parameters']
    assert not any(k in p for k in ('prompt', 'size', 'n'))


def test_kling_accepts_explicit_std_tier():
    p = make_adapter()._build_payload('x', tool='txt2video', resolution='std', duration=5)
    assert p['parameters']['mode'] == 'std'


def test_kling_rejects_duration_outside_model_range():
    with pytest.raises(ValueError, match='3～15'):
        make_adapter()._build_payload('x', tool='txt2video', duration=2)
    with pytest.raises(ValueError, match='3～15'):
        make_adapter()._build_payload('x', tool='txt2video', duration=16)


def test_kling_rejects_unsupported_ratio():
    with pytest.raises(ValueError, match='16:9/9:16/1:1'):
        make_adapter()._build_payload('x', tool='txt2video', duration=5, ratio='4:3')


def test_kling_drops_seed_with_warning():
    p = make_adapter()._build_payload('x', tool='txt2video', duration=5, seed=42)
    assert 'seed' not in p['parameters']


def test_kling_forwards_negative_prompt():
    p = make_adapter()._build_payload('x', tool='txt2video', duration=5, negative_prompt='模糊, 变形')
    assert p['input']['negative_prompt'] == '模糊, 变形'


# ── wan2.7 新协议分支 ───────────────────────────────────────────────────

def test_wan27_payload_uses_resolution_and_seed():
    p = make_wan27()._build_payload('一只小猫在月光下奔跑', tool='txt2video',
                                    resolution='1080p', ratio='3:4', duration=6,
                                    seed=42, negative_prompt='模糊')
    assert p['input'] == {'prompt': '一只小猫在月光下奔跑', 'negative_prompt': '模糊'}
    assert p['parameters'] == {
        'resolution': '1080P', 'ratio': '3:4', 'duration': 6, 'seed': 42,
    }
    assert not any(k in p for k in ('prompt', 'size', 'n'))


def test_wan27_preserves_duration_without_snapping():
    # 旧实现会把 6 snap 成 5、12 snap 成 10；现在必须原样发送
    assert make_wan27()._build_payload('x', tool='txt2video', duration=6)['parameters']['duration'] == 6
    assert make_wan27()._build_payload('x', tool='txt2video', duration=12)['parameters']['duration'] == 12


def test_wan27_rejects_duration_outside_model_range():
    with pytest.raises(ValueError, match='2～15'):
        make_wan27()._build_payload('x', tool='txt2video', duration=1)
    with pytest.raises(ValueError, match='2～15'):
        make_wan27()._build_payload('x', tool='txt2video', duration=16)


def test_wan27_rejects_unsupported_resolution():
    with pytest.raises(ValueError, match='720p/1080p'):
        make_wan27()._build_payload('x', tool='txt2video', resolution='480p')


def test_wan27_rejects_out_of_range_seed():
    with pytest.raises(ValueError, match='2147483647'):
        make_wan27()._build_payload('x', tool='txt2video', duration=5, seed=-1)


# ── wan2.6~2.1 旧协议分支（size 必须给具体像素值） ─────────────────────

def test_wan26_payload_uses_documented_size_table():
    p = make_wan26()._build_payload('x', tool='txt2video', resolution='720p', ratio='16:9',
                                    duration=8, seed=7)
    assert p['parameters'] == {'size': '1280*720', 'duration': 8, 'seed': 7}


def test_wan26_uses_3_4_size():
    p = make_wan26()._build_payload('x', tool='txt2video', resolution='1080p', ratio='3:4')
    assert p['parameters']['size'] == '1248*1632'


def test_wan25_only_allows_documented_durations():
    assert make_wan26(**{'model_name_or_path': 'wan2.5-t2v-preview'})._build_payload(
        'x', tool='txt2video', duration=10)['parameters']['duration'] == 10
    with pytest.raises(ValueError, match='5 或 10'):
        make_wan26(**{'model_name_or_path': 'wan2.5-t2v-preview'})._build_payload(
            'x', tool='txt2video', duration=6)


def test_wan22_duration_is_fixed():
    adapter = make_wan26(**{'model_name_or_path': 'wan2.2-t2v-plus'})
    assert adapter._build_payload('x', tool='txt2video', duration=5)['parameters']['duration'] == 5
    with pytest.raises(ValueError, match='固定为 5'):
        adapter._build_payload('x', tool='txt2video', duration=6)


def test_wan22_rejects_unknown_ratio_for_tier():
    adapter = make_wan26(**{'model_name_or_path': 'wan2.2-t2v-plus'})
    # 480p 档位官方只给了 16:9 / 9:16 / 1:1
    with pytest.raises(ValueError, match='16:9/9:16/1:1'):
        adapter._build_payload('x', tool='txt2video', resolution='480p', ratio='4:3')


def test_unknown_dashscope_video_model_is_rejected():
    with pytest.raises(ValueError, match='无法识别'):
        make_adapter(model_name_or_path='some-unknown-video-model')._build_payload(
            'x', tool='txt2video')


# ── 能力表 ──────────────────────────────────────────────────────────────

def test_family_table_covers_documented_models():
    names = {f['name'] for f in _DASHSCOPE_VIDEO_FAMILIES}
    assert names == {'wan2.7', 'wan2.6', 'wan2.5', 'wan2.2', 'wan2.1-t2v-turbo',
                     'wan2.1-t2v-plus', 'kling'}
    for family in _DASHSCOPE_VIDEO_FAMILIES:
        assert family['default_tier'] in family['tiers']
        assert family['duration'][0] in ('range', 'choice', 'fixed')
        for tier in family['tiers']:
            assert _dashscope_ratios(family, tier)


def test_family_lookup_by_prefix():
    assert _dashscope_video_family('wan2.7-t2v')['name'] == 'wan2.7'
    assert _dashscope_video_family('kling/kling-v3-turbo-video-generation')['name'] == 'kling'
    assert _dashscope_video_family('wanx2.1-t2v-turbo')['name'] == 'wan2.1-t2v-turbo'
    assert _dashscope_video_family('wan2.1-t2v-plus')['name'] == 'wan2.1-t2v-plus'


def test_legacy_size_table_matches_official_doc():
    assert _DASHSCOPE_LEGACY_SIZE_TABLE['480p'] == {'16:9': '832*480', '9:16': '480*832', '1:1': '624*624'}
    assert _DASHSCOPE_LEGACY_SIZE_TABLE['720p']['1:1'] == '960*960'
    assert _DASHSCOPE_LEGACY_SIZE_TABLE['1080p']['4:3'] == '1632*1248'
    assert _DASHSCOPE_LEGACY_SIZE_TABLE['1080p']['9:16'] == '1080*1920'


def test_dashscope_duration_helper():
    assert _dashscope_duration(_dashscope_video_family('wan2.7-t2v'), 'wan2.7-t2v', 2) == 2
    assert _dashscope_duration(_dashscope_video_family('wan2.7-t2v'), 'wan2.7-t2v', 15) == 15
    assert _dashscope_duration(_dashscope_video_family('wan2.5-t2v-preview'), 'wan2.5-t2v-preview', 5) == 5


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


def test_poll_budget_covers_real_generation_time():
    """实测 wan2.7-t2v 720P/2 秒片段耗时 739.5s（约 12.3 分钟，并发争抢会更久），
    官方口径"通常 1~5 分钟"在本环境不成立 —— 20 分钟上限会误杀，现取 45 分钟。"""
    assert DashScopeVideoAdapter.POLL_INTERVAL_SECONDS == 15
    assert DashScopeVideoAdapter.POLL_TIMEOUT_SECONDS >= 2700
    assert DashScopeVideoAdapter.POLL_TIMEOUT_SECONDS // DashScopeVideoAdapter.POLL_INTERVAL_SECONDS >= 120


def test_poll_timeout_error_carries_task_id_and_recheck_url():
    """超时必须留可复查线索：task_id 24 小时内有效，勿重复创建任务。"""
    import pytest
    running = fake_resp(200, {'output': {'task_id': 't1', 'task_status': 'RUNNING'}})
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=running):
        with pytest.raises(TimeoutError) as ei:
            make_adapter()._poll_async(
                {'output': {'task_id': 't1', 'task_status': 'PENDING'}}, 'http://post', {})
    msg = str(ei.value)
    assert 't1' in msg and '/tasks/t1' in msg and '24' in msg


def test_poll_non_2xx_logs_response_body_without_secrets(caplog):
    """可重试的非 2xx 必须回显响应体（脱敏后），且单次抖动不应终止任务。"""
    import logging
    body = '{"code":"InternalError","message":"upstream busy","token":"sk-abcdef123456"}'
    seq = [
        fake_resp(500, {}, text=body),
        fake_resp(200, {'output': {'task_id': 't1', 'task_status': 'SUCCEEDED', 'video_url': 'https://cdn/v.mp4'}}),
    ]
    with caplog.at_level(logging.WARNING), mock.patch('time.sleep'), mock.patch('requests.get', side_effect=seq):
        result = make_adapter()._poll_async(
            {'output': {'task_id': 't1', 'task_status': 'PENDING'}}, 'http://post', {})
    assert result['output']['task_status'] == 'SUCCEEDED'
    log = caplog.text
    assert 'HTTP 500' in log and 'upstream busy' in log
    assert 'sk-abcdef123456' not in log and '***' in log


def test_poll_fatal_403_fails_fast_without_retrying():
    """鉴权/权限类 403（实测：current user api does not support asynchronous calls）必须立即失败，
    不能每 15s 重试到 20 分钟超时 —— 错误里要带 message 与 request_id。"""
    import pytest
    body = ('{"code":"AccessDenied","message":"current user api does not support asynchronous calls",'
            '"request_id":"1d528790-3f9b-97a4-b2bf-3ea8f769c2c7"}')
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=fake_resp(403, {}, text=body)) as mg:
        with pytest.raises(RuntimeError) as ei:
            make_adapter()._poll_async(
                {'output': {'task_id': 't1', 'task_status': 'PENDING'}}, 'http://post', {})
    assert mg.call_count == 1  # 第一次就失败，没有空转
    msg = str(ei.value)
    assert 'does not support asynchronous calls' in msg
    assert '1d528790-3f9b-97a4-b2bf-3ea8f769c2c7' in msg and 't1' in msg


def test_poll_strips_dashscope_async_header():
    """查询任务是同步 GET：带 X-DashScope-Async 会被拒
    （403 AccessDenied「current user api does not support asynchronous calls」，官方 error-code 文档）。"""
    ok = fake_resp(200, {'output': {'task_id': 't1', 'task_status': 'SUCCEEDED', 'video_url': 'https://cdn/v.mp4'}})
    adapter = make_adapter()
    with mock.patch('time.sleep'), mock.patch('requests.get', return_value=ok) as mg:
        adapter._poll_async(
            {'output': {'task_id': 't1', 'task_status': 'PENDING'}}, 'http://post', adapter._build_headers())
    sent = mg.call_args.kwargs['headers']
    assert not any(k.lower() == 'x-dashscope-async' for k in sent), sent
    assert 'Authorization' in sent


def test_create_keeps_dashscope_async_header():
    """创建任务相反：缺该头会报 does not support synchronous calls，必须带 enable。"""
    assert make_adapter()._build_headers()['X-DashScope-Async'] == 'enable'


def test_extract_bytes_downloads_video():
    with mock.patch('requests.get', return_value=fake_resp(200, {}, content=b'REAL_MP4')) as mg:
        data = {'output': {'task_id': 't1', 'task_status': 'SUCCEEDED', 'video_url': 'https://cdn/v.mp4'}}
        assert make_adapter()._extract_bytes(data) == b'REAL_MP4'
        assert mg.call_args[0][0] == 'https://cdn/v.mp4'


def test_extract_bytes_missing_url_raises():
    import pytest
    with pytest.raises(ValueError, match='Cannot extract'):
        make_adapter()._extract_bytes({'output': {'task_id': 't1', 'task_status': 'PENDING'}})


def test_aspect_ratio_helper():
    assert _wh_to_aspect_ratio(1280, 720) == '16:9'
    assert _wh_to_aspect_ratio(854, 480) == '16:9'
    assert _wh_to_aspect_ratio(960, 720) == '4:3'
