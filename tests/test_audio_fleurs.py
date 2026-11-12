"""FLEURS AUDIO integration: no network and no real dataset required."""
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask

from evalscope.backend.audio_eval.arguments import AudioToolConfig
from evalscope.backend.audio_eval.backend_manager import AudioBackendManager
from evalscope.backend.audio_eval.datasets import fleurs
from evalscope.service.blueprints.audio import bp_audio


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('EVALSCOPE_CACHE', str(tmp_path))
    root = tmp_path / 'audio/fleurs/cmn_hans_cn/test'
    app = Flask(__name__)
    app.register_blueprint(bp_audio)
    return app.test_client(), root


def prepared(root, samples=None):
    root.mkdir(parents=True, exist_ok=True)
    audio = root / 'audio'
    audio.mkdir(exist_ok=True)
    for name in ('one.wav', 'two.wav'):
        with wave.open(str(audio / name), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 160)
    items = samples if samples is not None else [
        {'id': '001', 'text': '你好世界', 'audio': 'audio/one.wav'},
        {'id': '002', 'text': '今天天气好', 'audio': 'audio/two.wav'},
    ]
    (root / 'manifest.json').write_text(json.dumps({
        'dataset': 'google/fleurs', 'config': 'cmn_hans_cn', 'split': 'test',
        'revision': 'pinned-revision', 'status': 'ready', 'samples': items,
    }, ensure_ascii=False))


def config(out, sample='001'):
    return AudioToolConfig(tool='asr', model={'model_name_or_path': 'mock-asr'},
                           generate={'audio_source': 'fleurs', 'dataset_sample_id': sample,
                                     'reference_text': 'do not trust browser',
                                     'audio_base64': 'invalid and ignored'},
                           eval={'output_dir': str(out), 'metrics': ['wer', 'cer']})


def manager():
    obj = AudioBackendManager.__new__(AudioBackendManager)
    return obj


def test_unprepared_blocks_lookup_and_model_call(client, tmp_path):
    http, _ = client
    assert http.get('/api/v1/audio/fleurs/status').json['ready'] is False
    assert http.get('/api/v1/audio/fleurs/samples').json['samples'] == []
    assert http.get('/api/v1/audio/fleurs/random').status_code == 404
    obj = manager()
    obj._create_asr_model = lambda _: pytest.fail('model called before sample validation')
    with pytest.raises(FileNotFoundError, match='not prepared'):
        obj._run_asr(config(tmp_path / 'output'))


def test_list_search_random_preview_and_source_scoring(client, tmp_path):
    http, root = client
    prepared(root)
    assert http.get('/api/v1/audio/fleurs/status').json['sample_count'] == 2
    listing = http.get('/api/v1/audio/fleurs/samples?limit=1&offset=1').json
    assert listing['total'] == 2 and listing['samples'][0]['id'] == '002'
    assert http.get('/api/v1/audio/fleurs/samples?q=天气').json['samples'][0]['id'] == '002'
    assert http.get('/api/v1/audio/fleurs/random').json['id'] in {'001', '002'}
    sample = http.get('/api/v1/audio/fleurs/sample/001').json
    assert sample['text'] == '你好世界'
    assert http.get(sample['audio_url']).data.startswith(b'RIFF')
    seen = []
    obj = manager()
    obj._create_asr_model = lambda _: SimpleNamespace(generate=lambda path, **kwargs: seen.append(path) or {'text': '你好世界'})
    result = obj._run_asr(config(tmp_path / 'output'))
    assert seen == [str(root / 'audio/one.wav')]
    assert result['per_sample']['reference'] == '你好世界'
    assert result['metrics'] == {'wer': 0, 'cer': 0}
    assert result['dataset']['revision'] == 'pinned-revision'
    assert result['dataset']['sample_id'] == '001'
    assert json.loads((tmp_path / 'output/results.json').read_text())['dataset'] == result['dataset']


def test_missing_corrupt_and_escaped_audio_rejected_before_model(client, tmp_path):
    http, root = client
    obj = manager()
    obj._create_asr_model = lambda _: pytest.fail('invalid audio reached model')
    for path in ('audio/missing.wav', '../../escape.wav', 'audio/one.wav'):
        prepared(root, [{'id': '001', 'text': '你好', 'audio': path}])
        if path == 'audio/one.wav':
            (root / path).write_bytes(b'not audio')
        assert http.get('/api/v1/audio/fleurs/status').json['ready'] is False
        assert http.get('/api/v1/audio/fleurs/file/001').status_code == 404
        with pytest.raises(FileNotFoundError, match='not prepared'):
            obj._run_asr(config(tmp_path / 'output'))


def test_duplicate_and_malformed_ids_disable_dataset(client):
    http, root = client
    for bad in ([{'id': '001', 'text': 'a', 'audio': 'audio/one.wav'}] * 2,
                [{'id': '../escape', 'text': 'a', 'audio': 'audio/one.wav'}]):
        prepared(root, bad)
        assert http.get('/api/v1/audio/fleurs/status').json['ready'] is False
