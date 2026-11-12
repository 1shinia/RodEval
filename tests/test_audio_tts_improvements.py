"""Focused TTS regression tests; no provider calls."""
import json
import random
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest

from evalscope.backend.audio_eval.arguments import AudioToolConfig
from evalscope.backend.audio_eval.backend_manager import AudioBackendManager
from evalscope.backend.audio_eval.datasets.prompt_loader import TTS_BUILTIN, TTS_DATASET, load_tts_prompts


def manager():
    return AudioBackendManager.__new__(AudioBackendManager)


def config(tmp_path, **eval_options):
    return AudioToolConfig(tool='tts', model={'model_name_or_path': 'tts-1'},
                           eval={'output_dir': str(tmp_path), **eval_options},
                           generate={'response_format': 'wav'})


def wav_bytes():
    output = BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\x00\x00' * 1600)
    return output.getvalue()


def test_builtin_sampling_is_repeatable_with_same_seed_and_allows_explicit_choice(tmp_path):
    random.seed(1)
    first = load_tts_prompts(limit=3, seed=42)
    random.seed(999)
    assert load_tts_prompts(limit=3, seed=42) == first
    assert len(first) == 3
    assert load_tts_prompts(limit=3, selected_index=4, seed=42) == [TTS_BUILTIN[4]]
    with pytest.raises(ValueError, match='index'):
        load_tts_prompts(limit=3, selected_index=100, seed=42)


def test_dataset_categories_and_selection_are_consistent():
    items = TTS_DATASET['items']
    assert len(items) == len(TTS_BUILTIN) == 100
    assert [int(item['id']) for item in items] == list(range(1, 101))
    assert len({item['text'] for item in items}) == 100
    assert [sum(item['category'] == category for item in items) for category in dict.fromkeys(item['category'] for item in items)] == [20, 20, 15, 15, 15, 15]
    category = items[20]['category']
    chosen = load_tts_prompts(limit=2, seed=9, category=category)
    assert chosen == load_tts_prompts(limit=2, seed=9, category=category)
    assert all(text in [item['text'] for item in items if item['category'] == category] for text in chosen)
    assert load_tts_prompts(selected_index=20, category=category) == [items[20]['text']]
    with pytest.raises(ValueError, match='category'):
        load_tts_prompts(selected_index=0, category=category)


def test_selected_prompt_and_provenance_are_written_to_report(tmp_path):
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: wav_bytes())
    result = obj._run_tts(config(tmp_path, selected_prompt_index=4, prompt_limit=7))
    assert [s['prompt'] for s in result['per_sample']] == [TTS_BUILTIN[4]]
    assert result['prompt_source'] == 'builtin'
    assert result['prompt_seed'] == 42
    assert result['selected_prompt_index'] == 4
    assert result['prompt_dataset_version'] == 'tts-basic-100-v1'
    assert result['per_sample'][0]['prompt_id'] == '005'
    assert result['per_sample'][0]['prompt_category'] == TTS_DATASET['items'][4]['category']


def test_category_filter_reaches_pipeline(tmp_path):
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: wav_bytes())
    category = TTS_DATASET['items'][20]['category']
    result = obj._run_tts(config(tmp_path, prompt_limit=2, prompt_category=category))
    assert result['prompt_category'] == category
    assert all(sample['prompt'] in load_tts_prompts(limit=20, category=category) for sample in result['per_sample'])


@pytest.mark.parametrize('payload', [b'', b'<html>error</html>', b'RIFF' + b'\x00' * 4 + b'WAVEbroken'])
def test_invalid_audio_never_counts_as_success(tmp_path, payload):
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: payload)
    with pytest.raises(RuntimeError, match='all samples failed'):
        obj._run_tts(config(tmp_path, custom_prompt='test'))
    result = json.loads((tmp_path / 'failed_results.json').read_text())
    assert 'error' in result['per_sample'][0]
    assert not list((tmp_path / 'audio').glob('*'))


def test_closed_loop_reports_partial_failure_coverage(tmp_path):
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: wav_bytes())
    calls = iter([{'text': 'hello'}, RuntimeError('asr unavailable')])
    def transcribe(*a, **kw):
        response = next(calls)
        if isinstance(response, Exception):
            raise response
        return response
    obj._create_asr_model = lambda _: SimpleNamespace(generate=transcribe)
    result = obj._run_tts(config(tmp_path, custom_prompt=None, prompt_limit=2, asr_model_name='asr-1', metrics=['wer']))
    assert result['coverage']['planned'] == 2
    assert result['coverage']['tts_success'] == 2
    assert result['coverage']['asr_success'] == 1
    assert result['coverage']['wer_scored'] == 1
    assert result['per_sample'][1]['asr_error'] == 'asr unavailable'
    assert result['asr_model'] == 'asr-1'
    assert result['status'] == 'partial'


def test_all_asr_failures_preserve_audio_and_closed_loop_identity(tmp_path):
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: wav_bytes())
    def fail(*a, **kw):
        raise RuntimeError('recognition unavailable')
    obj._create_asr_model = lambda _: SimpleNamespace(generate=fail)
    result = obj._run_tts(config(tmp_path, custom_prompt='hello', asr_model_name='asr', metrics=['wer']))
    assert result['coverage']['tts_success'] == 1
    assert result['coverage']['asr_success'] == 0
    assert result['coverage']['wer_scored'] == 0
    assert result['closed_loop_enabled']
    assert result['asr_model'] == 'asr'
    assert result['status'] == 'partial'
    assert 'metrics' not in result


def test_scoring_failure_excluded_from_mean(tmp_path, monkeypatch):
    from evalscope.backend.audio_eval.metrics import wer
    monkeypatch.setattr(wer, 'compute_wer', lambda *a: -1.0)
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: wav_bytes())
    obj._create_asr_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: {'text': 'hello'})
    result = obj._run_tts(config(tmp_path, custom_prompt='hello', asr_model_name='asr', metrics=['wer']))
    assert result['coverage']['asr_success'] == 1
    assert result['coverage']['wer_scored'] == 0
    assert result['status'] == 'partial'
    assert 'metrics' not in result


@pytest.mark.parametrize('options', [{'asr_model_name': ' '}, {'asr_model_name': 'asr', 'metrics': []}])
def test_invalid_closed_loop_config_rejected(tmp_path, options):
    obj = manager()
    obj._create_tts_model = lambda _: SimpleNamespace(generate=lambda *a, **kw: pytest.fail('must not call provider'))
    with pytest.raises(ValueError):
        obj._run_tts(config(tmp_path, **options))
