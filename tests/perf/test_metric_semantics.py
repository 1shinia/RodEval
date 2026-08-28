"""Regression tests for benchmark metric semantics hardened in schema v2."""

import math

from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.db_util import calculate_percentiles


class _Plugin:
    @staticmethod
    def parse_responses(_messages, request=None):
        return 10, 1


def test_one_token_response_has_no_tpot():
    data = BenchmarkData(
        success=True,
        query_latency=0.20,
        first_chunk_latency=0.02,
        first_token_latency=0.10,
        prompt_tokens=10,
        completion_tokens=1,
    )
    data.finalize(_Plugin())
    assert data.time_per_output_token is None


def test_tpot_uses_first_generated_token_not_first_sse_chunk():
    data = BenchmarkData(
        success=True,
        query_latency=0.30,
        first_chunk_latency=0.02,  # e.g. role-only SSE event
        first_token_latency=0.10,
        prompt_tokens=10,
        completion_tokens=3,
    )
    data.finalize(_Plugin())
    assert math.isclose(data.time_per_output_token, 0.10)


def test_ttft_average_excludes_successful_zero_token_responses():
    acc = MetricsAccumulator()
    with_token = BenchmarkData(
        success=True, start_time=0.0, completed_time=0.2, query_latency=0.2,
        first_chunk_latency=0.02, first_token_latency=0.10,
        prompt_tokens=10, completion_tokens=1,
    )
    zero_token = BenchmarkData(
        success=True, start_time=0.0, completed_time=0.1, query_latency=0.1,
        first_chunk_latency=0.01, first_token_latency=None,
        prompt_tokens=10, completion_tokens=0,
    )
    acc.update(with_token, _Plugin())
    acc.update(zero_token, _Plugin())
    assert math.isclose(acc.to_result().avg_first_token_latency, 0.10)



def test_zero_token_response_has_no_ttft_even_if_adapter_set_nonstream_latency():
    data = BenchmarkData(
        success=True,
        query_latency=0.20,
        first_chunk_latency=0.20,
        first_token_latency=0.20,
        prompt_tokens=10,
        completion_tokens=0,
    )
    data.finalize(_Plugin())
    assert data.first_token_latency is None
    assert data.time_per_output_token is None

def test_percentiles_use_linear_interpolation_not_off_by_one_indexing():
    values = list(range(100))
    result = calculate_percentiles(values, [50, 99, 100])
    assert result[50] == 49.5
    assert result[99] == 98.01
    assert result[100] == 99.0
