"""文生图专用契约；网络使用替身，不调用付费接口。"""
import base64
import pytest
from pathlib import Path
from unittest import mock

from evalscope.backend.aigc_eval.models.adapters import OpenAICompatibleAdapter, create_adapter

FORM_PATH = Path(__file__).resolve().parents[2] / 'evalscope/web/src/components/eval/AIGCEvalForm.tsx'


def adapter():
    return OpenAICompatibleAdapter({'model_name_or_path': 'image-model', 'model_type': 'txt2img'})


def test_txt2img_prompt_dataset_dropdown_only_contains_builtin_image_sets():
    source = FORM_PATH.read_text(encoding='utf-8')
    block = source[source.index('const TEXT_TO_IMAGE_DATASETS'):source.index('const IMAGE_DATASETS')]

    assert "{ value: 'drawbench', label: 'DrawBench (200 prompts)' }" in block
    assert "{ value: 'parti', label: 'PartiPrompts (150 prompts)' }" in block
    assert "value: 'custom'" not in block
    assert "value: ''" not in block
    assert "prompt_dataset: tool !== 'txt2video' && promptDataset !== 'parti' ? 'drawbench' : promptDataset" in source
    assert 'evalConfig.custom_dataset_path' not in source


@pytest.mark.parametrize('message', ['invalid model', 'size is not supported', 'invalid resolution'])
def test_image_errors_never_guess_a_different_size(message):
    response = mock.Mock(status_code=400, text=message)
    payload = adapter()._build_payload('cat', tool='txt2img', width=1024, height=1024)
    with mock.patch('requests.post') as post:
        with pytest.raises(ValueError, match=message):
            adapter()._handle_retry(response, '', {}, payload, tool='txt2img')
    post.assert_not_called()
    assert payload['size'] == '1024x1024'


@pytest.mark.parametrize('provider', ['volcengine'])
def test_txt2img_rejects_video_only_providers(provider):
    with pytest.raises(ValueError, match='文生图'):
        create_adapter({'provider': provider, 'model_type': 'txt2img'})


@pytest.mark.parametrize('width', [0, 1024.5, True, 'wrong'])
def test_invalid_image_dimensions_are_explicit_errors(width):
    with pytest.raises(ValueError, match='尺寸'):
        adapter()._build_payload('cat', tool='txt2img', width=width, height=1024)


def test_image_request_evidence_excludes_prompt_and_records_duration():
    a = adapter()
    response = mock.Mock(ok=True)
    response.json.return_value = {'data': [{'b64_json': base64.b64encode(b'png').decode()}]}
    with mock.patch('requests.post', return_value=response):
        assert a.generate('private prompt', tool='txt2img', width=1024, height=1024, seed=0) == b'png'
    assert a.last_image_request['parameters'] == {'model': 'image-model', 'n': 1, 'size': '1024x1024', 'seed': 0}
    assert a.last_image_request['parameter_support'] == 'unconfirmed'
    assert a.last_image_request['request_seconds'] >= 0


@pytest.mark.parametrize('tool', ['txt2video', 'img2img'])
def test_other_tools_keep_original_defaults(tool):
    from evalscope.backend.aigc_eval.arguments import AIGCToolConfig
    config = AIGCToolConfig(tool=tool, model={'model_name_or_path': 'm', 'api_base': 'https://example.test'})
    if tool == 'img2img':
        assert config.generate.num_inference_steps is None
        assert config.generate.guidance_scale is None
        assert config.generate.seed is None
    else:
        assert config.generate.num_inference_steps == 50
        assert config.generate.guidance_scale == 7.5
        assert config.generate.seed == 42


def test_image_api_defaults_are_optional_but_explicit_values_are_preserved():
    from evalscope.backend.aigc_eval.arguments import AIGCToolConfig
    model = {'model_name_or_path': 'm', 'api_base': 'https://example.test'}
    config = AIGCToolConfig(model=model)
    assert config.generate.num_inference_steps is None
    assert config.generate.guidance_scale is None
    assert config.generate.seed is None
    explicit = AIGCToolConfig(model=model, generate={'seed': 0, 'num_inference_steps': 20})
    assert explicit.generate.seed == 0
    assert explicit.generate.num_inference_steps == 20


def test_image_pipeline_saves_request_and_actual_dimensions(tmp_path):
    import json
    from io import BytesIO
    from PIL import Image

    from evalscope.backend.aigc_eval.backend_manager import AIGCBackendManager
    buf = BytesIO()
    Image.new('RGB', (320, 256)).save(buf, format='PNG')
    response = mock.Mock(ok=True)
    response.json.return_value = {'data': [{'b64_json': base64.b64encode(buf.getvalue()).decode()}]}
    config = {
        'tool': 'txt2img',
        'model': {'model_name_or_path': 'm', 'api_base': 'https://example.test/v1'},
        'eval': {'output_dir': str(tmp_path), 'custom_prompt': 'cat', 'metrics': []},
    }
    with mock.patch('requests.post', return_value=response) as post:
        result = AIGCBackendManager(config).run()
    payload = post.call_args.kwargs['json']
    assert 'seed' not in payload
    assert 'num_inference_steps' not in payload
    assert 'guidance_scale' not in payload
    sample = result['per_sample'][0]
    assert (sample['actual_width'], sample['actual_height']) == (320, 256)
    assert sample['api_request']['parameters']['size'] == '512x512'
    assert sample['image_generation_seconds'] >= sample['api_request']['request_seconds']
    assert json.loads((tmp_path / 'results.json').read_text()) == result
