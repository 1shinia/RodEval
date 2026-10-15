"""Offline regression for the single-reference image editing chain."""
import base64
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from evalscope.backend.aigc_eval.arguments import AIGCToolConfig
from evalscope.backend.aigc_eval.models.adapters import create_adapter
from evalscope.backend.aigc_eval.models.img2img import Img2ImgModel


def png(size=(640, 512)):
    buf = io.BytesIO()
    Image.new('RGB', size, 'red').save(buf, format='PNG')
    return buf.getvalue()


def config(tmp_path, **generate):
    return AIGCToolConfig(tool='img2img', model={'model_name_or_path': 'qwen-image-2.0',
        'provider': 'dashscope', 'api_base': 'https://dashscope.aliyuncs.com/api/v1'},
        generate=generate, eval={'output_dir': str(tmp_path), 'custom_prompt': '改成蓝色',
        'reference_image': base64.b64encode(png()).decode(), 'metrics': []})


def test_edit_dispatch_and_payload(tmp_path):
    cfg = config(tmp_path)
    adapter = create_adapter({**cfg.model.model_dump(), 'tool': 'img2img'})
    assert type(adapter).__name__ == 'DashScopeImageEditAdapter'
    payload = adapter._build_payload('改成蓝色', reference_image_b64=base64.b64encode(png()).decode(),
        width=1024, height=1024, seed=0)
    content = payload['input']['messages'][0]['content']
    assert content[0]['image'].startswith('data:image/png;base64,')
    assert content[1] == {'text': '改成蓝色'}
    assert payload['parameters'] == {'size': '1024*1024', 'n': 1, 'seed': 0}
    assert 'X-DashScope-Async' not in adapter._build_headers()
    assert adapter._resolve_url('img2img').endswith('/services/aigc/multimodal-generation/generation')


def test_strength_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        config(tmp_path, strength=0.8)


def test_api_optional_defaults(tmp_path):
    cfg = config(tmp_path)
    assert cfg.generate.seed is None
    assert cfg.generate.num_inference_steps is None
    assert cfg.generate.guidance_scale is None



def test_model_preserves_reference_and_forwards_parameters():
    model = Img2ImgModel({'model_name_or_path': 'gateway', 'tool': 'img2img'})
    model._adapter = Mock(last_image_request={})
    model._adapter.generate.return_value = png()
    model.generate(['edit'], reference_images=[Image.new('RGB', (640, 512), 'red')],
        width=1024, height=1024, num_inference_steps=13, guidance_scale=3,
        negative_prompt='noise', seed=0)
    kwargs = model._adapter.generate.call_args.kwargs
    reference = Image.open(io.BytesIO(base64.b64decode(kwargs['reference_image_b64'])))
    assert reference.size == (640, 512)
    assert kwargs['num_inference_steps'] == 13
    assert kwargs['guidance_scale'] == 3
    assert kwargs['negative_prompt'] == 'noise'


def test_image_provider_options_exclude_video_only_provider():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2]
              / 'evalscope/web/src/components/eval/AIGCEvalForm.tsx').read_text()
    assert "PROVIDER_OPTIONS.filter(p => tool === 'txt2video' || p.value === 'openai' || p.value === 'dashscope')" in source


def test_gateway_omits_strength():
    adapter = create_adapter({'tool': 'img2img', 'provider': 'openai', 'model_name_or_path': 'gateway'})
    payload = adapter._build_payload('edit', tool='img2img',
        num_inference_steps=13, guidance_scale=3, seed=0,
        reference_image_b64=base64.b64encode(png()).decode())
    assert 'strength' not in payload
    assert payload['num_inference_steps'] == 13
    assert payload['guidance_scale'] == 3
    assert payload['seed'] == 0
    assert payload['image'].startswith('data:image/png;base64,')
    assert 'image' not in adapter.last_image_request['parameters']


def test_gateway_does_not_add_strength_default(tmp_path):
    cfg = AIGCToolConfig.model_validate({
        'tool': 'img2img', 'model': {
            'model_name_or_path': 'gateway', 'provider': 'openai', 'api_base': 'https://example.com/v1',
        }, 'eval': {'output_dir': str(tmp_path)},
    })
    assert not hasattr(cfg.generate, 'strength')
    assert cfg.generate.num_inference_steps is None
    assert cfg.generate.guidance_scale is None
    assert cfg.generate.seed is None


def test_service_omits_gateway_strength(tmp_path, monkeypatch):
    from evalscope.service.blueprints import aigc
    monkeypatch.setattr(aigc, 'OUTPUT_DIR', tmp_path)
    for provider in ['openai', 'dashscope']:
        payload = {
            'tool': 'img2img', 'model': {'provider': provider, 'api_base': 'https://example.com/api/v1'},
            'generate': {'num_inference_steps': 13, 'guidance_scale': 3, 'seed': 0},
        }
        built = aigc._build_aigc_config(payload, provider)
        parsed = AIGCToolConfig.model_validate(built)
        assert not hasattr(parsed.generate, 'strength')
        assert parsed.generate.num_inference_steps == 13
        assert parsed.generate.guidance_scale == 3
        assert parsed.generate.seed == 0


def test_img2img_excludes_dataset_controls_and_payload():
    source = (Path(__file__).resolve().parents[2] / 'evalscope/web/src/components/eval/AIGCEvalForm.tsx').read_text()
    controls = source.split("{tool !== 'img2img' && (<>", 1)[1].split('</>)}', 1)[0]
    assert "t('aigc.promptDataset')" in controls
    assert "t('aigc.promptLimit')" in controls
    assert 'checked={randomPrompt}' in controls
    assert "t('aigc.metrics')" not in controls
    payload = source.split("if (tool !== 'img2img') {", 1)[1].split('    if (referenceImageBase64)', 1)[0]
    for field in ('prompt_dataset:', 'prompt_limit:', 'random_prompt:'):
        assert field in payload
    assert 'evalConfig.custom_prompt = customPrompt.trim()' in source
    assert 'evalConfig.reference_image = referenceImageBase64' in source
    assert 'evalConfig.custom_dataset_path' not in source


def test_img2img_form_provider_branches():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] /
              'evalscope/web/src/components/eval/AIGCEvalForm.tsx').read_text()
    branch = source[source.index("if (tool === 'img2img') {"):source.index('    setErrors({})', source.index("if (tool === 'img2img') {"))]
    assert "if (provider === 'dashscope' && !/^qwen-image" in branch
    assert "if (provider === 'dashscope' && (!apiBase" in branch
    assert "newErrors.provider = '图生图首期仅支持" not in branch
    assert 'strength' not in source
    assert "num_inference_steps: provider === 'dashscope' ? '' : imageSteps" in source
    assert "guidance_scale: provider === 'dashscope' ? '' : imageGuidance" in source


@pytest.mark.parametrize('reference', [None, 'invalid', base64.b64encode(b'not image').decode()])
def test_bad_reference_fails_before_model(tmp_path, monkeypatch, reference):
    from evalscope.backend.aigc_eval.backend_manager import AIGCBackendManager
    cfg = config(tmp_path)
    cfg.eval.reference_image = reference
    load = Mock()
    monkeypatch.setattr(Img2ImgModel, 'load', load)
    with pytest.raises(ValueError):
        AIGCBackendManager(cfg.model_dump())._run_img2img(cfg)
    load.assert_not_called()


def test_metric_failure_keeps_report(tmp_path, monkeypatch):
    import sys
    from evalscope.backend.aigc_eval.backend_manager import AIGCBackendManager
    metric = Mock()
    metric.compute_clip_score.side_effect = RuntimeError('weights unavailable')
    monkeypatch.setitem(sys.modules, 'evalscope.backend.aigc_eval.metrics.clip_score', metric)
    monkeypatch.setattr(Img2ImgModel, 'load', lambda self: None)
    monkeypatch.setattr(Img2ImgModel, 'generate', lambda *args, **kwargs: [Image.new('RGB', (704, 512))])
    cfg = config(tmp_path)
    cfg.eval.metrics = ['clip_score']
    result = AIGCBackendManager(cfg.model_dump())._run_img2img(cfg)
    saved = json.loads((tmp_path / 'results.json').read_text())
    assert saved == result
    assert result['reference_path'] == 'images/reference.png'
    assert result['reference_size'] == [640, 512]
    assert result['per_sample'][0]['output_size'] == [704, 512]
    assert 'clip_score_mean' not in result['metrics']
    assert result['metrics']['clip_score_valid_samples'] == 0
    assert result['per_sample'][0]['metric_errors']['clip_score']


@pytest.mark.parametrize('params', [
    {'num_inference_steps': 10}, {'guidance_scale': 3},
    {'width': 256, 'height': 256}, {'seed': -1}, {'seed': True},
])
def test_qwen_rejects_unsupported_or_invalid_before_post(tmp_path, monkeypatch, params):
    import evalscope.backend.aigc_eval.models.adapters as adapters
    adapter = create_adapter({**config(tmp_path).model.model_dump(), 'tool': 'img2img'})
    post = Mock()
    monkeypatch.setattr(adapters.requests, 'post', post)
    with pytest.raises(ValueError):
        adapter.generate('edit', tool='img2img', reference_image_b64=base64.b64encode(png()).decode(), **params)
    post.assert_not_called()


def test_qwen_http_error_not_retried(tmp_path, monkeypatch):
    import evalscope.backend.aigc_eval.models.adapters as adapters
    adapter = create_adapter({**config(tmp_path).model.model_dump(), 'tool': 'img2img'})
    post = Mock(return_value=Mock(ok=False, status_code=400, text='InvalidParameter'))
    monkeypatch.setattr(adapters.requests, 'post', post)
    with pytest.raises(ValueError, match='HTTP 400'):
        adapter.generate('edit', tool='img2img', reference_image_b64=base64.b64encode(png()).decode())
    assert post.call_count == 1


def test_qwen_response_download(tmp_path, monkeypatch):
    import evalscope.backend.aigc_eval.models.adapters as adapters
    adapter = create_adapter({**config(tmp_path).model.model_dump(), 'tool': 'img2img'})
    post = Mock(return_value=Mock(ok=True, json=lambda: {
        'output': {'choices': [{'message': {'content': [{'image': 'https://example.test/result.png'}]}}]}}))
    get = Mock(return_value=Mock(content=png()))
    monkeypatch.setattr(adapters.requests, 'post', post)
    monkeypatch.setattr(adapters.requests, 'get', get)
    assert adapter.generate('edit', tool='img2img', reference_image_b64=base64.b64encode(png()).decode()) == png()
    assert post.call_count == 1
    get.assert_called_once_with('https://example.test/result.png', timeout=120)


def test_qwen_optional_flags(tmp_path):
    adapter = create_adapter({**config(tmp_path).model.model_dump(), 'tool': 'img2img'})
    payload = adapter._build_payload('edit', reference_image_b64=base64.b64encode(png()).decode(),
                                     prompt_extend=False, watermark=False)
    assert payload['parameters']['prompt_extend'] is False
    assert payload['parameters']['watermark'] is False
    assert not any('base64' in str(v) for v in adapter.last_image_request['parameters'].values())


def test_metric_success_records_count(tmp_path, monkeypatch):
    import sys
    from evalscope.backend.aigc_eval.backend_manager import AIGCBackendManager
    metric = Mock(CLIP_SCORE_META={'kind': 'text relevance'})
    metric.compute_clip_score.return_value = [0.75]
    monkeypatch.setitem(sys.modules, 'evalscope.backend.aigc_eval.metrics.clip_score', metric)
    monkeypatch.setattr(Img2ImgModel, 'load', lambda self: None)
    monkeypatch.setattr(Img2ImgModel, 'generate', lambda *a, **k: [Image.new('RGB', (512, 512))])
    cfg = config(tmp_path)
    cfg.eval.metrics = ['clip_score', 'lpips']
    result = AIGCBackendManager(cfg.model_dump())._run_img2img(cfg)
    assert result['metrics']['clip_score_mean'] == 0.75
    assert result['metrics']['clip_score_valid_samples'] == 1
    assert 'lpips_mean' not in result['metrics']
    assert result['per_sample'][0]['metric_errors']['lpips']


def test_blank_instruction_before_loading(tmp_path, monkeypatch):
    from evalscope.backend.aigc_eval.backend_manager import AIGCBackendManager
    cfg = config(tmp_path)
    cfg.eval.custom_prompt = ' '
    load = Mock()
    monkeypatch.setattr(Img2ImgModel, 'load', load)
    with pytest.raises(ValueError, match='编辑指令'):
        AIGCBackendManager(cfg.model_dump())._run_img2img(cfg)
    load.assert_not_called()


def test_alias_reference_not_saved():
    adapter = create_adapter({'tool': 'img2img', 'provider': 'custom', 'model_name_or_path': 'm',
        'param_aliases': {'image': 'source_image'}})
    adapter._build_payload('edit', tool='img2img', reference_image_b64=base64.b64encode(png()).decode())
    assert 'source_image' not in adapter.last_image_request['parameters']


def test_reference_media_respects_access(tmp_path, monkeypatch):
    from flask import Flask
    import evalscope.service.blueprints.aigc as aigc
    import evalscope.service.blueprints.auth as auth
    root = tmp_path / 'outputs'
    image = root / 'task_01' / 'images' / 'reference.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(png())
    monkeypatch.setattr(aigc, 'OUTPUT_DIR', root)
    app = Flask(__name__)
    with app.test_request_context('/file/task_01/images/reference.png'):
        monkeypatch.setattr(auth, 'check_task_artifact_access', lambda *a, **k: False)
        result = aigc.serve_file('task_01', 'images/reference.png')
        assert (result[1] if isinstance(result, tuple) else result.status_code) == 404
        monkeypatch.setattr(auth, 'check_task_artifact_access', lambda *a, **k: True)
        response = aigc.serve_file('task_01', 'images/reference.png')
        assert response.status_code == 200
        response.close()


