# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for benchmark description payload modes.

The benchmark list endpoint serialises the whole catalogue in one response, and
README bodies dominate that payload (~89% of ~4 MB measured).  ``build_benchmark_entry``
therefore takes a ``description_mode`` so list consumers can truncate or skip
READMEs, while the detail endpoint still returns them in full.
"""
import json

import pytest

from evalscope.service.utils.benchmarks import (
    _DESCRIPTION_PREVIEW_CHARS,
    build_benchmark_entry,
    discover_all_benchmarks,
)

#: Everything except the description itself must survive every mode unchanged.
_DESCRIPTION_KEYS = ('description', 'description_truncated')


def _without_description(entry):
    return {k: v for k, v in entry.items() if k not in _DESCRIPTION_KEYS}


def _benchmark_with_long_readme():
    """Return a catalogue name whose Chinese README exceeds one preview."""
    for name in discover_all_benchmarks():
        zh = (build_benchmark_entry(name).get('description') or {}).get('zh') or {}
        if len(zh.get('full') or '') > _DESCRIPTION_PREVIEW_CHARS:
            return name
    pytest.skip('no benchmark README long enough to exercise preview truncation')


def test_default_mode_returns_complete_readme():
    name = _benchmark_with_long_readme()
    entry = build_benchmark_entry(name)
    assert entry['description_truncated'] is False
    assert len(entry['description']['zh']['full']) > _DESCRIPTION_PREVIEW_CHARS


def test_preview_truncates_and_flags_entry():
    entry = build_benchmark_entry(_benchmark_with_long_readme(), 'preview')
    zh = entry['description']['zh']
    assert entry['description_truncated'] is True
    assert len(zh['full']) == _DESCRIPTION_PREVIEW_CHARS
    assert zh['sections'] == {}


def test_none_omits_description_but_keeps_metadata():
    name = _benchmark_with_long_readme()
    entry = build_benchmark_entry(name, 'none')
    assert entry['description'] == {}
    assert entry['description_truncated'] is False
    assert entry['name'] == name
    assert entry['meta'], 'metadata must survive description suppression'


@pytest.mark.parametrize('mode', ['preview', 'none'])
def test_only_description_fields_differ(mode):
    """Every non-description field must be byte-identical to the full entry."""
    name = _benchmark_with_long_readme()
    assert _without_description(build_benchmark_entry(name, mode)) == _without_description(build_benchmark_entry(name))


def test_unknown_mode_falls_back_to_full():
    """The endpoint rejects unknown modes with 400; the builder must not crash."""
    name = _benchmark_with_long_readme()
    assert build_benchmark_entry(name, 'bogus') == build_benchmark_entry(name, 'full')


def test_flag_is_set_exactly_when_preview_loses_content():
    """An unset flag would silently render a partial README, so pin both ways."""
    for name in discover_all_benchmarks():
        preview = build_benchmark_entry(name, 'preview')
        whole = build_benchmark_entry(name)['description']
        cut = any(
            (preview['description'].get(lang) or {}).get('full') != (whole.get(lang) or {}).get('full')
            for lang in ('zh', 'en')
        )
        assert preview['description_truncated'] == cut, name


def test_preview_cuts_catalogue_payload_below_half():
    names = discover_all_benchmarks()

    def payload(mode):
        return sum(len(json.dumps(build_benchmark_entry(n, mode), ensure_ascii=False).encode()) for n in names)

    full, preview = payload('full'), payload('preview')
    assert preview < full * 0.5, f'preview {preview} B not <50% of full {full} B'
