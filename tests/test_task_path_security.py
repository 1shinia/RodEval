"""Regression tests for task identifier and artifact path boundaries."""

import os

import pytest

from evalscope.service.utils.log import resolve_task_dir, resolve_task_file, validate_task_id


@pytest.mark.parametrize('task_id', [
    '', '.', '..', '../etc', 'a/b', r'a\\b', '/absolute', '\x00task',
    'task id', 'task?id', 'task#id', 'a' * 256,
])
def test_validate_task_id_rejects_path_or_ambiguous_identifiers(task_id):
    with pytest.raises(ValueError):
        validate_task_id(task_id)


@pytest.mark.parametrize('task_id', [
    'eval_1782000000000',
    'perf_dltest_001',
    'aigc-run.01',
    'audio-task_01',
])
def test_validate_task_id_preserves_compatible_identifiers(task_id):
    validate_task_id(task_id)


def test_resolve_task_dir_requires_direct_child_of_output_root(tmp_path):
    root = tmp_path / 'outputs'
    root.mkdir()
    task_dir = root / 'eval_ok'
    task_dir.mkdir()

    assert resolve_task_dir('eval_ok', str(root), must_exist=True) == task_dir.resolve()

    with pytest.raises(ValueError):
        resolve_task_dir('.', str(root))
    with pytest.raises(ValueError):
        resolve_task_dir('..', str(root))
    with pytest.raises(ValueError):
        resolve_task_dir('../outside', str(root))


def test_resolve_task_dir_rejects_symlinked_task_directory(tmp_path):
    root = tmp_path / 'outputs'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'eval_link').symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        resolve_task_dir('eval_link', str(root))


def test_resolve_task_dir_rejects_alias_to_another_task(tmp_path):
    root = tmp_path / 'outputs'
    root.mkdir()
    target = root / 'eval_real'
    target.mkdir()
    (root / 'eval_alias').symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError):
        resolve_task_dir('eval_alias', str(root))


def test_resolve_task_file_rejects_internal_files_and_symlink_escape(tmp_path):
    root = tmp_path / 'outputs'
    task_dir = root / 'eval_ok'
    media_dir = task_dir / 'media'
    media_dir.mkdir(parents=True)
    (task_dir / '.owner').write_text('1')
    (task_dir / 'task_config.yaml').write_text('secret: no')
    (media_dir / 'ok.png').write_bytes(b'ok')
    outside = tmp_path / 'outside.txt'
    outside.write_text('outside')
    (media_dir / 'escape.txt').symlink_to(outside)

    assert resolve_task_file('eval_ok', 'media/ok.png', str(root), allowed_subdirs=('media',)) == (media_dir / 'ok.png').resolve()

    for filename in ['.owner', 'task_config.yaml', '../.owner', 'media/../.owner', 'media/escape.txt']:
        with pytest.raises(ValueError):
            resolve_task_file('eval_ok', filename, str(root), allowed_subdirs=('media',))

    with pytest.raises(ValueError):
        resolve_task_file('eval_ok', 'logs/run.log', str(root), allowed_subdirs=('media',))
