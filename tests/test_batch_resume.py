"""Tests for credential-free batch manifests and resume row selection."""

from evalscope.service.batch_resume import batch_manifest_hash, resumable_row_indexes


def test_manifest_ignores_credentials_but_detects_execution_changes():
    rows_a = [
        {'model': 'm1', 'base_url': 'http://x', 'api_key': 'secret-a', 'concurrency': '1'},
        {'model': 'm1', 'base_url': 'http://y', 'api_key': 'secret-b', 'concurrency': '2'},
    ]
    rows_b = [dict(rows_a[0], api_key='rotated-a'), dict(rows_a[1], api_key='rotated-b')]
    shared = {'dataset': 'openqa', 'headers': {'Authorization': 'Bearer old'}}
    rotated = {'dataset': 'openqa', 'headers': {'Authorization': 'Bearer new'}}

    assert batch_manifest_hash(rows_a, shared) == batch_manifest_hash(rows_b, rotated)
    assert batch_manifest_hash(rows_a, shared) != batch_manifest_hash(
        [dict(rows_a[0], concurrency='4'), rows_a[1]], shared
    )
    assert batch_manifest_hash(rows_a, shared) != batch_manifest_hash(list(reversed(rows_a)), shared)


def test_resume_only_runs_pending_and_interrupted_rows():
    items = [
        {'row_index': 0, 'status': 'completed'},
        {'row_index': 1, 'status': 'failed'},
        {'row_index': 2, 'status': 'interrupted'},
        {'row_index': 3, 'status': 'pending'},
    ]

    assert resumable_row_indexes(items) == {2, 3}
