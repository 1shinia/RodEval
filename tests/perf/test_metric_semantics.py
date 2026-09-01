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


def test_tpot_excludes_protocol_tail_after_last_generated_token():
    data = BenchmarkData(
        success=True,
        start_time=100.0,
        completed_time=100.50,      # includes usage/DONE tail
        last_generated_time=100.30, # last actual generated token
        query_latency=0.50,
        first_token_latency=0.10,
        prompt_tokens=10,
        completion_tokens=3,
    )
    data.finalize(_Plugin())
    assert math.isclose(data.time_per_output_token, 0.10)


def test_sse_chunk_count_is_not_used_as_speculative_decode_telemetry():
    data = BenchmarkData(
        success=True,
        start_time=1.0,
        completed_time=1.4,
        query_latency=0.4,
        first_token_latency=0.1,
        prompt_tokens=10,
        completion_tokens=9,
        chunk_times=[1.1, 1.2, 1.3],
    )
    data.finalize(_Plugin())
    assert data.decoded_tokens_per_iter is None


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


def test_percentiles_keep_full_precision_in_data_layer():
    result = calculate_percentiles([0.004, 0.006], [50])
    assert result[50] == 0.005


def test_invalid_zero_timestamp_failure_does_not_corrupt_wall_clock():
    acc = MetricsAccumulator()
    acc.update(BenchmarkData(
        success=True, start_time=10.0, completed_time=11.0,
        query_latency=1.0, prompt_tokens=10, completion_tokens=5,
    ), _Plugin())
    acc.update(BenchmarkData(success=False), _Plugin())
    assert math.isclose(acc.wall_time, 1.0)


def test_timed_failure_extends_wall_clock_and_reduces_goodput():
    acc = MetricsAccumulator()
    acc.update(BenchmarkData(
        success=True, start_time=10.0, completed_time=11.0,
        query_latency=1.0, prompt_tokens=10, completion_tokens=5,
    ), _Plugin())
    acc.update(BenchmarkData(
        success=False, start_time=10.5, completed_time=15.0,
        query_latency=4.5,
    ), _Plugin())
    result = acc.to_result()
    assert math.isclose(acc.wall_time, 5.0)
    assert math.isclose(result.qps, 1 / 5)


def test_embedding_input_throughput_uses_concurrent_workload_wall_time():
    # Embedding/rerank workloads have no completion tokens; input throughput
    # must be computed against the concurrent wall-clock window, not the sum of
    # per-request latencies.  MetricsAccumulator computes input throughput
    # against wall time unconditionally, so no api_type discriminator is needed.
    acc = MetricsAccumulator()
    acc.update(BenchmarkData(
        success=True, start_time=10.0, completed_time=11.0,
        query_latency=1.0, prompt_tokens=1000, completion_tokens=0,
    ), _Plugin())
    acc.update(BenchmarkData(
        success=True, start_time=10.0, completed_time=11.0,
        query_latency=1.0, prompt_tokens=1000, completion_tokens=0,
    ), _Plugin())
    result = acc.to_result()
    assert math.isclose(result.avg_input_token_throughput, 2000.0)
