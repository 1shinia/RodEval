"""Offline contract tests for native Volcengine audio APIs."""
import base64
from unittest.mock import Mock, patch

import pytest

from evalscope.backend.audio_eval.backend_manager import AudioBackendManager
from evalscope.backend.audio_eval.arguments import AudioToolConfig
from evalscope.backend.audio_eval.models.asr_volcengine import AsrModelVolcengine
from evalscope.backend.audio_eval.models.tts_volcengine import TtsModelVolcengine


BASE = 'https://openspeech.bytedance.com'


def response(status=200, headers=None, body=None, lines=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = body or {}
    result.text = 'upstream error'
    result.iter_lines.return_value = lines or []
    return result


def test_asr_uploads_base64_using_flash_protocol(tmp_path):
    audio = tmp_path / 'input.mp3'
    audio.write_bytes(b'ID3test')
    model = AsrModelVolcengine({'model_name_or_path': 'volc.bigasr.auc_turbo', 'api_base': BASE, 'api_key': 'secret'})
    with patch('evalscope.backend.audio_eval.models.asr_volcengine.requests.post', return_value=response(
        headers={'X-Api-Status-Code': '20000000'}, body={'result': {'text': '测试成功'}}
    )) as post:
        assert model.generate(str(audio), language='zh')['text'] == '测试成功'
    args, kwargs = post.call_args
    assert args[0] == BASE + '/api/v3/auc/bigmodel/recognize/flash'
    assert kwargs['headers']['X-Api-Resource-Id'] == 'volc.bigasr.auc_turbo'
    assert kwargs['headers']['X-Api-Key'] == 'secret'
    assert kwargs['headers']['X-Api-Sequence'] == '-1'
    assert kwargs['json']['audio']['data'] == base64.b64encode(audio.read_bytes()).decode('ascii')
    assert kwargs['json']['audio']['format'] == 'mp3'
    assert kwargs['json']['request']['model_name'] == 'bigmodel'
    assert 'url' not in kwargs['json']['audio']
    assert kwargs['json']['audio']['language'] == 'zh-CN'


def test_asr_rejects_wrong_resource_or_format_before_request(tmp_path):
    audio = tmp_path / 'input.m4a'
    audio.write_bytes(b'fake')
    with patch('evalscope.backend.audio_eval.models.asr_volcengine.requests.post') as post:
        with pytest.raises(ValueError):
            AsrModelVolcengine({'model_name_or_path': 'volc.seedasr.auc', 'api_base': BASE, 'api_key': 'secret'}).generate(str(audio))
        with pytest.raises(ValueError):
            AsrModelVolcengine({'model_name_or_path': 'volc.bigasr.auc_turbo', 'api_base': BASE, 'api_key': 'secret'}).generate(str(audio))
    post.assert_not_called()


def test_asr_rejects_upstream_failure(tmp_path):
    audio = tmp_path / 'input.wav'
    audio.write_bytes(b'RIFFtest')
    model = AsrModelVolcengine({'model_name_or_path': 'volc.bigasr.auc_turbo', 'api_base': BASE, 'api_key': 'secret'})
    with patch('evalscope.backend.audio_eval.models.asr_volcengine.requests.post', return_value=response(
        headers={'X-Api-Status-Code': '45000000', 'X-Api-Message': 'invalid resource'}
    )):
        with pytest.raises(RuntimeError, match='45000000'):
            model.generate(str(audio))


def test_tts_decodes_stream_chunks():
    model = TtsModelVolcengine({'model_name_or_path': 'seed-tts-2.0', 'api_base': BASE, 'api_key': 'secret'})
    chunks = [b'{"code":0,"data":"YWI="}', b'{"code":0,"data":"Y2Q="}', b'{"code":0,"message":"OK"}']
    with patch('evalscope.backend.audio_eval.models.tts_volcengine.requests.post', return_value=response(lines=chunks)) as post:
        assert model.generate('你好', voice='zh_female_vv_uranus_bigtts', response_format='mp3', speed=1.0) == b'abcd'
    args, kwargs = post.call_args
    assert args[0] == BASE + '/api/v3/tts/unidirectional'
    assert kwargs['headers']['X-Api-Resource-Id'] == 'seed-tts-2.0'
    assert kwargs['headers']['X-Api-Key'] == 'secret'
    assert kwargs['json'] == {'req_params': {'text': '你好', 'speaker': 'zh_female_vv_uranus_bigtts', 'audio_params': {'format': 'mp3'}}}
    assert kwargs['stream'] is True


def test_tts_rejects_invalid_speed_voice_and_stream_error():
    model = TtsModelVolcengine({'model_name_or_path': 'seed-tts-2.0', 'api_base': BASE, 'api_key': 'secret'})
    with patch('evalscope.backend.audio_eval.models.tts_volcengine.requests.post') as post:
        with pytest.raises(ValueError, match='voice'):
            model.generate('你好', voice='alloy')
        with pytest.raises(ValueError, match='speed'):
            model.generate('你好', voice='zh_female_vv_uranus_bigtts', speed=3)
        post.assert_not_called()
    with patch('evalscope.backend.audio_eval.models.tts_volcengine.requests.post', return_value=response(
        lines=[b'{"code":0,"data":"YWI="}', b'{"code":3001,"message":"failed"}']
    )):
        with pytest.raises(RuntimeError, match='3001'):
            model.generate('你好', voice='zh_female_vv_uranus_bigtts')


def test_factory_routes_volcengine_and_closed_loop_provider_is_independent():
    manager = AudioBackendManager({})
    assert isinstance(manager._create_asr_model({'provider': 'volcengine', 'model_name_or_path': 'volc.bigasr.auc_turbo'}), AsrModelVolcengine)
    assert isinstance(manager._create_tts_model({'provider': 'volcengine', 'model_name_or_path': 'seed-tts-2.0'}), TtsModelVolcengine)


def test_closed_loop_config_accepts_independent_provider():
    config = AudioToolConfig.model_validate({
        'tool': 'tts', 'model': {'model_name_or_path': 'tts-1', 'provider': 'openai'},
        'eval': {'asr_model_name': 'volc.bigasr.auc_turbo', 'asr_provider': 'volcengine'},
    })
    assert config.eval.asr_provider == 'volcengine'


def test_tts_ogg_uses_native_format_and_preserves_speed():
    model = TtsModelVolcengine({'model_name_or_path': 'seed-tts-2.0', 'api_base': BASE, 'api_key': 'secret'})
    with patch('evalscope.backend.audio_eval.models.tts_volcengine.requests.post', return_value=response(
        lines=[b'{"code":0,"data":"YWI="}', b'{"code":20000000}']
    )) as post:
        assert model.generate('hello', voice='speaker-id', response_format='ogg', speed=1.5) == b'ab'
    params = post.call_args.kwargs['json']['req_params']['audio_params']
    assert params == {'format': 'ogg_opus', 'speech_rate': 50}
