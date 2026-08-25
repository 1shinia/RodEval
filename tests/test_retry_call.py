# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for retry_call error classification.

Vendor verifiers (kimi/minimax/k2) probe that the API *rejects* non-default
immutable parameters with HTTP 400 — an expected signal that must not be
retried.  Only transient errors (timeout, 5xx, 429) should retry.
"""
from evalscope.utils.function_utils import _is_retryable_error, retry_call


class _HTTPError(Exception):
    def __init__(self, code):
        super().__init__(f'http {code}')
        self.status_code = code


def test_expected_400_is_not_retried():
    calls = []

    def boom():
        calls.append(1)
        raise _HTTPError(400)

    try:
        retry_call(boom, retries=5, sleep_interval=0)
        raise AssertionError('expected 400 to propagate')
    except _HTTPError:
        pass
    assert len(calls) == 1, '4xx must not be retried'


def test_429_is_retried():
    calls = []

    def boom():
        calls.append(1)
        raise _HTTPError(429)

    try:
        retry_call(boom, retries=3, sleep_interval=0)
        raise AssertionError('expected 429 to exhaust retries')
    except _HTTPError:
        pass
    assert len(calls) == 3, '429 is rate-limit -> retryable'


def test_5xx_is_retried():
    calls = []

    def boom():
        calls.append(1)
        raise _HTTPError(503)

    try:
        retry_call(boom, retries=3, sleep_interval=0)
        raise AssertionError('expected 503 to exhaust retries')
    except _HTTPError:
        pass
    assert len(calls) == 3


def test_network_error_is_retried():
    calls = []

    def boom():
        calls.append(1)
        raise TimeoutError('read timed out')

    try:
        retry_call(boom, retries=3, sleep_interval=0)
        raise AssertionError('expected timeout to exhaust retries')
    except TimeoutError:
        pass
    assert len(calls) == 3


def test_classification_table():
    assert _is_retryable_error(_HTTPError(400)) is False
    assert _is_retryable_error(_HTTPError(401)) is False
    assert _is_retryable_error(_HTTPError(404)) is False
    assert _is_retryable_error(_HTTPError(429)) is True
    assert _is_retryable_error(_HTTPError(500)) is True
    assert _is_retryable_error(_HTTPError(503)) is True
    assert _is_retryable_error(TimeoutError()) is True
