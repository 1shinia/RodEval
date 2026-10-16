import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalscope.backend.audio_eval.arguments import AudioToolConfig
from evalscope.backend.audio_eval.backend_manager import AudioBackendManager
from evalscope.backend.audio_eval.metrics.wer import compute_cer, compute_wer
from evalscope.backend.audio_eval.datasets.prompt_loader import TTS_BUILTIN, load_tts_prompts
from evalscope.service.blueprints.audio import bp_audio


def manager(config):
    obj = AudioBackendManager.__new__(AudioBackendManager)
    obj._audio_config = config
    return obj


def asr_config(tmp_path, filename='clip.wav', content=b'RIFF', reference='hello world'):
    return AudioToolConfig(
        tool='asr',
        model={'model_name_or_path': 'whisper-1'},
        generate={'audio_base64': base64.b64encode(content).decode(),
                  'audio_filename': filename, 'reference_text': reference},
        eval={'output_dir': str(tmp_path), 'metrics': ['wer']},
    )


@pytest.mark.parametrize('filename', ['../escaped.wav', '/tmp/escaped.wav', 'sub/clip.wav', '..\\escaped.wav'])
def test_asr_rejects_unsafe_filename(tmp_path, filename):
    with pytest.raises(ValueError, match='filename'):
        manager(asr_config(tmp_path, filename))._run_asr(asr_config(tmp_path, filename))


def test_asr_rejects_oversize_audio_before_writing(tmp_path, monkeypatch):
    import evalscope.backend.audio_eval.backend_manager as module
    monkeypatch.setattr(module, 'MAX_AUDIO_BYTES', 3, raising=False)
    config = asr_config(tmp_path, content=b'1234')
    with pytest.raises(ValueError, match='size'):
        manager(config)._run_asr(config)
    assert not (tmp_path / 'audio' / 'clip.wav').exists()


def test_asr_rejects_invalid_encoding(tmp_path):
    config = asr_config(tmp_path)
    config.generate.audio_base64 = 'not valid base64'
    with pytest.raises(ValueError, match='encoding'):
        manager(config)._run_asr(config)


def test_asr_without_reference_does_not_report_zero_error(tmp_path):
    config = asr_config(tmp_path, reference='')
    obj = manager(config)
    obj._create_asr_model = lambda _: SimpleNamespace(generate=lambda *a, **k: {'text': ''})
    result = obj._run_asr(config)
    assert 'wer' not in result['metrics']


def tts_config(tmp_path, custom_prompt: str | None = 'hello', prompt_source=None, prompt_limit=1):
    eval_config = {'output_dir': str(tmp_path), 'prompt_limit': prompt_limit}
    if custom_prompt is not None:
        eval_config['custom_prompt'] = custom_prompt
    if prompt_source is not None:
        eval_config['prompt_source'] = prompt_source
    return AudioToolConfig(
        tool='tts', model={'model_name_or_path': 'tts-1'}, eval=eval_config,
    )


def test_tts_all_samples_failed_is_failure(tmp_path):
    config = tts_config(tmp_path)
    obj = manager(config)
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')))
    with pytest.raises(RuntimeError, match='all.*failed'):
        obj._run_tts(config)
    result = json.loads((tmp_path / 'failed_results.json').read_text())
    assert result['per_sample'][0]['error'] == 'offline'
    assert not (tmp_path / 'results.json').exists()


def test_tts_success_is_not_failure(tmp_path, monkeypatch):
    import evalscope.backend.audio_eval.utils.audio as audio_utils
    monkeypatch.setattr(audio_utils, 'get_audio_duration', lambda _: 1.0)
    config = tts_config(tmp_path)
    obj = manager(config)
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **k: b'audio')
    result = obj._run_tts(config)
    assert len(result['per_sample']) == 1
    assert 'error' not in result['per_sample'][0]


def test_tts_custom_prompt_overrides_builtin_dataset_by_default(tmp_path, monkeypatch):
    from evalscope.backend.audio_eval.datasets import prompt_loader
    from evalscope.backend.audio_eval.utils import audio as audio_utils

    monkeypatch.setattr(prompt_loader, 'load_tts_prompts', lambda *args, **kwargs: pytest.fail('dataset should not load'))
    monkeypatch.setattr(audio_utils, 'get_audio_duration', lambda _: 1.0)
    config = tts_config(tmp_path, custom_prompt='  one custom text  ', prompt_limit=10)
    obj = manager(config)
    generated = []
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda prompt, **kwargs: generated.append(prompt) or b'audio')

    result = obj._run_tts(config)

    assert generated == ['one custom text']
    assert [sample['prompt'] for sample in result['per_sample']] == generated


def test_tts_explicit_builtin_source_uses_dataset_even_if_custom_text_is_present(tmp_path, monkeypatch):
    from evalscope.backend.audio_eval import backend_manager as module
    from evalscope.backend.audio_eval.datasets import prompt_loader
    from evalscope.backend.audio_eval.utils import audio as audio_utils

    prompts = TTS_BUILTIN
    monkeypatch.setattr(prompt_loader, 'load_tts_prompts', lambda dataset, limit, custom_path: prompts[:limit])
    monkeypatch.setattr(audio_utils, 'get_audio_duration', lambda _: 1.0)
    generated = []
    config = tts_config(tmp_path, custom_prompt='must not replace the builtin source', prompt_source='builtin', prompt_limit=10)
    obj = manager(config)
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda prompt, **kwargs: generated.append(prompt) or b'audio')

    result = obj._run_tts(config)

    assert generated == TTS_BUILTIN
    assert [sample['prompt'] for sample in result['per_sample']] == generated


def test_tts_custom_source_requires_text(tmp_path):
    config = tts_config(tmp_path, custom_prompt=None, prompt_source='custom')
    with pytest.raises(ValueError, match='custom.*prompt'):
        manager(config)._run_tts(config)


def test_tts_prompt_endpoint_returns_builtin_list():
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp_audio)
    response = app.test_client().get('/api/v1/audio/tts-prompts')

    assert response.status_code == 200
    assert response.get_json() == {'prompts': TTS_BUILTIN}
    assert len(response.get_json()['prompts']) == 10


def test_builtin_tts_sampling_ignores_task_seed(monkeypatch):
    import random
    from evalscope.backend.audio_eval.datasets import prompt_loader

    class DistinctSampler:
        calls = 0

        def sample(self, population, count):
            self.calls += 1
            return [population[self.calls]]

    sampler = DistinctSampler()
    monkeypatch.setattr(random, 'SystemRandom', lambda: sampler)
    random.seed(42)
    first = prompt_loader.load_tts_prompts(limit=1)
    random.seed(42)
    second = prompt_loader.load_tts_prompts(limit=1)

    assert first != second
    assert first == [TTS_BUILTIN[1]]
    assert second == [TTS_BUILTIN[2]]


def test_custom_dataset_requires_path():
    with pytest.raises(ValueError, match='custom.*path'):
        load_tts_prompts('custom', custom_path=None)


def test_english_wer_retains_word_boundary():
    assert compute_wer('hello world', 'hello earth') == pytest.approx(0.5)
    assert compute_wer('hello world', 'helloworld') > 0
    assert compute_wer('hello,world', 'hello world') == pytest.approx(0.0)
    assert compute_cer('hello world', 'helloworld') == pytest.approx(0.0)


def test_mixed_language_wer_retains_english_words():
    assert compute_wer('你好 hello world', '你好 hello earth') == pytest.approx(0.25)
