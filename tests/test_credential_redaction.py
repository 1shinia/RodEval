"""Regression tests for credential redaction in serialized benchmark arguments."""

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments


def test_perf_arguments_redact_nested_and_variant_sensitive_keys():
    secret = 'perf-secret-sentinel'
    args = Arguments(
        model='m1',
        api_key=secret,
        headers={'authorization': f'Bearer {secret}', 'X-API-Key': secret},
        extra_args={'nested': {'access-token': secret, 'max_tokens': 128}},
    )

    dumped = args.to_dict()

    assert secret not in repr(dumped)
    assert dumped['headers']['authorization'] == '***'
    assert dumped['headers']['X-API-Key'] == '***'
    assert dumped['extra_args']['nested']['access-token'] == '***'
    assert dumped['extra_args']['nested']['max_tokens'] == 128


def test_task_config_redacts_nested_judge_credentials():
    secret = 'judge-secret-sentinel'
    config = TaskConfig(
        model='m1',
        api_key=secret,
        judge_model_args={'api_key': secret, 'headers': {'Authorization': f'Bearer {secret}'}},
    )

    dumped = config.to_dict()

    assert secret not in repr(dumped)
    assert dumped['judge_model_args']['api_key'] == '***'
    assert dumped['judge_model_args']['headers']['Authorization'] == '***'


def test_resume_credentials_never_reuses_redacted_disk_values():
    from evalscope.service.blueprints.eval import _inject_resume_credentials

    restored = _inject_resume_credentials(
        {'api_key': '***', 'judge_model_args': {'api_key': '***', 'model_id': 'judge'}}, {}
    )
    assert 'api_key' not in restored
    assert 'api_key' not in restored['judge_model_args']


def test_resume_credentials_are_injected_only_from_request():
    from evalscope.service.blueprints.eval import _inject_resume_credentials

    restored = _inject_resume_credentials(
        {'api_key': '***', 'judge_model_args': {'api_key': '***', 'model_id': 'judge'}},
        {'api_key': 'model-secret', 'judge_model_args': {'api_key': 'judge-secret'}},
    )
    assert restored['api_key'] == 'model-secret'
    assert restored['judge_model_args']['api_key'] == 'judge-secret'


def test_resume_default_judge_reuses_current_request_model_key():
    from evalscope.service.blueprints.eval import _inject_resume_credentials

    restored = _inject_resume_credentials(
        {'api_key': '***', 'judge_model_args': {'api_key': '***', 'model_id': 'model'}},
        {'api_key': 'current-model-secret'},
    )

    assert restored['api_key'] == 'current-model-secret'
    assert restored['judge_model_args']['api_key'] == 'current-model-secret'


def test_resume_explicit_judge_key_takes_priority_over_model_key():
    from evalscope.service.blueprints.eval import _inject_resume_credentials

    restored = _inject_resume_credentials(
        {'api_key': '***', 'judge_model_args': {'api_key': '***', 'model_id': 'judge'}},
        {'api_key': 'current-model-secret', 'judge_model_args': {'api_key': 'current-judge-secret'}},
    )

    assert restored['api_key'] == 'current-model-secret'
    assert restored['judge_model_args']['api_key'] == 'current-judge-secret'
