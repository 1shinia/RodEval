from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin


def _request(**kwargs):
    args = Arguments(model='demo-model', api='openai', **kwargs)
    return OpenaiPlugin(args).build_request([{'role': 'user', 'content': 'hello'}])


def test_min_tokens_auto_enables_ignore_eos():
    request = _request(min_tokens=128)

    assert request['min_tokens'] == 128
    assert request['ignore_eos'] is True


def test_explicit_ignore_eos_override_is_preserved():
    assert _request(min_tokens=128, extra_args={'ignore_eos': False})['ignore_eos'] is False
    assert _request(min_tokens=128, extra_args={'ignore_eos': True})['ignore_eos'] is True


def test_missing_min_tokens_does_not_add_ignore_eos():
    assert 'ignore_eos' not in _request()