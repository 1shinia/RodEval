# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for eval report sample-count computation.

``Report.num`` already collapses multiple metrics inside one benchmark by
using the maximum metric sample count.  ``_compute_total_num`` therefore works
at the next level up: each report entry represents a benchmark/dataset and
positive counts must be summed across reports.
"""
from types import SimpleNamespace

from evalscope.report.report import Category, Metric, Report, Subset
from evalscope.service.db import _compute_total_num


def _r(num):
    return SimpleNamespace(num=num)


def test_single_report_limited():
    assert _compute_total_num([_r(1)]) == 1


def test_single_report_full_count():
    assert _compute_total_num([_r(1319)]) == 1319


def test_multiple_benchmarks_sum_sample_counts():
    assert _compute_total_num([_r(100), _r(200)]) == 300


def test_zero_count_reports_do_not_reduce_total():
    assert _compute_total_num([_r(0), _r(5), _r(0)]) == 5


def test_all_zero_falls_back_to_full_sentinel():
    assert _compute_total_num([_r(0), _r(0)]) == -1


def test_mixed_negative_and_positive():
    assert _compute_total_num([_r(-1), _r(5)]) == 5


def test_empty_list():
    assert _compute_total_num([]) == -1


# --- Report.num: max across metrics within a single benchmark ---


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
