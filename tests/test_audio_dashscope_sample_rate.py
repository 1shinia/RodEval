"""DashScope WebSocket ASR must declare the input audio's real sample rate."""
import json
import shutil
import subprocess
import wave
from contextlib import nullcontext

import pytest

from evalscope.backend.audio_eval.models import asr_dashscope_ws


class FakeWS:
    def __init__(self):
        self.sent = []
        self.responses = iter([
            json.dumps({'header': {'event': 'task-started'}}),
            json.dumps({'header': {'event': 'result-generated'},
                        'payload': {'output': {'sentence': {'text': 'hello'}}}}),
            json.dumps({'header': {'event': 'task-finished'}}),
        ])

    def send(self, data):
        self.sent.append(data)

    def recv(self, timeout):
        return next(self.responses)


def run_asr(path, monkeypatch):
    ws = FakeWS()
    monkeypatch.setattr(asr_dashscope_ws, 'ws_connect', lambda *a, **kw: nullcontext(ws))
    model = asr_dashscope_ws.AsrModelDashScopeWS({
        'model_name_or_path': 'paraformer-realtime-v2', 'api_base': 'https://example.com/api/v1',
        'api_key': 'test',
    })
    result = model.generate(str(path))
    return result, json.loads(ws.sent[0])['payload']['parameters']


def test_mp3_32000_hz_is_declared_to_asr(tmp_path, monkeypatch):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required to create MP3 fixture')
    wav_path = tmp_path / 'clip.wav'
    with wave.open(str(wav_path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(b'\0\0' * 3200)
    mp3_path = tmp_path / 'clip.mp3'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(wav_path), str(mp3_path)], check=True)
    result, params = run_asr(mp3_path, monkeypatch)
    assert result == {'text': 'hello'}
    assert params['format'] == 'mp3'
    assert params['sample_rate'] == 32000


def test_wav_24000_hz_is_declared_to_asr(tmp_path, monkeypatch):
    wav_path = tmp_path / 'clip.wav'
    with wave.open(str(wav_path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b'\0\0' * 2400)
    _, params = run_asr(wav_path, monkeypatch)
    assert params['format'] == 'wav'
    assert params['sample_rate'] == 24000


def test_invalid_mp3_fails_before_websocket(tmp_path, monkeypatch):
    invalid = tmp_path / 'broken.mp3'
    invalid.write_bytes(b'invalid audio')
    monkeypatch.setattr(asr_dashscope_ws, 'ws_connect', lambda *a, **kw: pytest.fail('WebSocket must not connect'))
    model = asr_dashscope_ws.AsrModelDashScopeWS({'model_name_or_path': 'paraformer-realtime-v2'})
    with pytest.raises(ValueError, match='sample rate'):
        model.generate(str(invalid))
