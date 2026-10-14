"""DashScope 图片协议回归；不调用付费接口。"""
import pytest
from unittest import mock

from evalscope.backend.aigc_eval.models.adapters import DashScopeVideoAdapter, create_adapter


def adapter(model='qwen-image-2.0-pro', **config):
    return create_adapter({'provider': 'dashscope', 'model_type': 'txt2img',
                           'model_name_or_path': model, 'api_base': 'https://dashscope.aliyuncs.com/api/v1',
                           **config})


@pytest.mark.parametrize('model', ['qwen-image-2.0-pro', 'wan2.6-t2i'])
def test_sync_request_and_download(model):
    a = adapter(model)
    post_response = mock.Mock(ok=True)
    post_response.json.return_value = {'output': {'choices': [{'message': {'content': [
        {'image': 'https://example.test/image.png'}]}}]}}
    download = mock.Mock(content=b'png')
    with mock.patch('requests.post', return_value=post_response) as post, \
            mock.patch('requests.get', return_value=download) as get:
        assert a.generate('private prompt', tool='txt2img', width=1280, height=1280, seed=0) == b'png'
    assert post.call_args.args[0].endswith('/services/aigc/multimodal-generation/generation')
    payload = post.call_args.kwargs['json']
    assert payload['input']['messages'][0]['content'] == [{'text': 'private prompt'}]
    assert payload['parameters'] == {'size': '1280*1280', 'n': 1, 'seed': 0}
    assert 'X-DashScope-Async' not in post.call_args.kwargs['headers']
    assert 'private prompt' not in str(a.last_image_request)
    assert a.last_image_request['request_seconds'] >= 0
    assert 'headers' not in get.call_args.kwargs


def test_wan_legacy_poll_and_download():
    a = adapter('wan2.2-t2i-plus')
    created = mock.Mock(ok=True)
    created.json.return_value = {'output': {'task_id': 'task-1', 'task_status': 'PENDING'}}
    finished = mock.Mock(ok=True)
    finished.json.return_value = {'output': {'task_status': 'SUCCEEDED',
                                          'results': [{'url': 'https://example.test/image.png'}]}}
    download = mock.Mock(content=b'png')
    with mock.patch('requests.post', return_value=created) as post, \
            mock.patch('requests.get', side_effect=[finished, download]) as get, \
            mock.patch('evalscope.backend.aigc_eval.models.adapters._time.sleep'):
        assert a.generate('cat', tool='txt2img', width=1024, height=1024) == b'png'
    assert post.call_args.args[0].endswith('/text2image/image-synthesis')
    assert post.call_args.kwargs['headers']['X-DashScope-Async'] == 'enable'
    assert post.call_args.kwargs['json']['input'] == {'prompt': 'cat'}
    assert get.call_args_list[0].args[0] == 'https://dashscope.aliyuncs.com/api/v1/tasks/task-1'


@pytest.mark.parametrize('params', [{'seed': -1}, {'seed': True}, {'seed': 2147483648},
                                  {'num_inference_steps': 20}, {'guidance_scale': 7.5},
                                  {'width': 0}, {'width': 512, 'height': 512}])
def test_invalid_parameters_fail_before_network(params):
    with mock.patch('requests.post') as post, pytest.raises(ValueError):
        adapter('wan2.6-t2i').generate('cat', tool='txt2img', **params)
    post.assert_not_called()


def test_unknown_model_and_compatible_mode_are_explicit_errors():
    with pytest.raises(ValueError, match='模型'):
        adapter('unknown-model')
    with pytest.raises(ValueError, match='compatible-mode'):
        adapter(api_base='https://dashscope.aliyuncs.com/compatible-mode/v1').generate('cat')


def test_failed_task_and_http_errors_do_not_resubmit():
    a = adapter('wan2.2-t2i-plus')
    created = mock.Mock(ok=True)
    created.json.return_value = {'output': {'task_id': 'task-1', 'task_status': 'FAILED',
                                          'code': 'BadRequest', 'message': 'bad prompt'}}
    with mock.patch('requests.post', return_value=created) as post, pytest.raises(RuntimeError, match='BadRequest'):
        a.generate('cat', width=1024, height=1024)
    assert post.call_count == 1
    rejected = mock.Mock(ok=False, status_code=400, text='InvalidParameter')
    with mock.patch('requests.post', return_value=rejected) as post, pytest.raises(ValueError, match='InvalidParameter'):
        a.generate('cat', width=1024, height=1024)
    assert post.call_count == 1


def test_video_factory_still_uses_video_adapter():
    assert isinstance(create_adapter({'provider': 'dashscope', 'model_type': 'txt2video'}), DashScopeVideoAdapter)
