# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for eval report sample-count computation.

Multi-metric reports (vendor verifiers, etc.) carry one row per metric whose
``num`` is the count of rows that triggered that validator (0 = not
triggered).  ``_compute_total_num`` must return the max positive count
instead of summing (over-count) or treating 0 rows as "full dataset"
(-1 sentinel).
"""
from types import SimpleNamespace

from evalscope.report.report import Category, Metric, Report, Subset
from evalscope.service.db import _compute_total_num


def _r(num):
    return SimpleNamespace(num=num)


def test_single_metric_limited():
    assert _compute_total_num([_r(1)]) == 1


def test_single_metric_full():
    assert _compute_total_num([_r(1319)]) == 1319


def test_multi_metric_with_untouched_validators():
    # minimax_verifier-style report: 4 validators not triggered (0), 2 with 1 row.
    report = [_r(0), _r(0), _r(1), _r(1), _r(0), _r(0)]
    assert _compute_total_num(report) == 1


def test_all_zero_falls_back_to_full_sentinel():
    assert _compute_total_num([_r(0), _r(0)]) == -1


def test_mixed_negative_and_positive():
    assert _compute_total_num([_r(-1), _r(5)]) == 5


def test_empty_list():
    assert _compute_total_num([]) == -1


# --- Report.num: max across metrics (vendor-verifier / multi_if cases) ---


def _report(metric_nums):
    """Build a Report whose metrics each have one category with one subset
    carrying the given sample count."""
    metrics = [
        Metric(name=f'm{i}', categories=[Category(name=('default',), subsets=[Subset(name='s', num=n, score=0.0)])])
        for i, n in enumerate(metric_nums)
    ]
    return Report(metrics=metrics)


def test_report_num_first_metric_zero():
    # minimax_verifier-style: first metric untriggered (0), later ones have rows.
    assert _report([0, 0, 1, 1, 0, 0]).num == 1


def test_report_num_multi_metric_same_samples():
    # multi_if-style: every metric counts the same sample set -> max == any.
    assert _report([8, 8, 8]).num == 8


def test_report_num_single_metric():
    assert _report([1319]).num == 1319


def test_report_num_no_metrics():
    assert Report(metrics=[]).num == 0
