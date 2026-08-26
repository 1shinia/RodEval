"""Regression tests for benchmark_data.db storage hardening."""
import base64
import json
import pickle
import sqlite3

from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.perf.utils.db_util import create_result_table, decode_data, insert_benchmark_data


def test_result_table_persists_diagnostics_as_json(tmp_path):
    db_path = tmp_path / 'benchmark_data.db'
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    create_result_table(cur)

    data = BenchmarkData(
        request=json.dumps({'model': 'm', 'prompt': 'hello'}),
        start_time=1.0,
        completed_time=1.5,
        success=False,
        response_messages=[json.dumps({'error': 'rate limited'})],
        error='HTTP 429',
        status_code=429,
        trace_id='trace-1',
        input_num_turns=2,
        is_last_turn=True,
        cached_tokens=64,
    )
    insert_benchmark_data(cur, data)
    con.commit()

    row = con.execute(
        '''SELECT request_id, response_messages, status_code, error, trace_id,
                  turn_index, cached_tokens, is_last_turn
           FROM result'''
    ).fetchone()
    assert row[0] == data.request_id
    assert decode_data(row[1]) == data.response_messages
    assert row[2:] == (429, 'HTTP 429', 'trace-1', 2, 64, 1)

    indexes = {r[1] for r in con.execute("PRAGMA index_list('result')")}
    assert {'idx_result_request_id', 'idx_result_success_start', 'idx_result_trace_turn'} <= indexes
    con.close()


def test_create_result_table_upgrades_legacy_shape(tmp_path):
    db_path = tmp_path / 'legacy.db'
    con = sqlite3.connect(db_path)
    con.execute(
        '''CREATE TABLE result(
            request TEXT, start_time REAL, inter_token_latencies TEXT, success INTEGER,
            response_messages TEXT, completed_time REAL, latency REAL,
            first_chunk_latency REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
            max_gpu_memory_cost REAL, time_per_output_token REAL
        )'''
    )
    create_result_table(con.cursor())
    columns = {r[1] for r in con.execute('PRAGMA table_info(result)')}
    assert {'request_id', 'status_code', 'error', 'trace_id', 'turn_index', 'cached_tokens'} <= columns
    con.close()


def test_json_storage_round_trips_binary_chunks():
    payload = ['text', b'\x00\xff', {'chunk': bytearray(b'abc')}]
    from evalscope.perf.utils.db_util import encode_data

    encoded = encode_data(payload)
    decoded = decode_data(encoded)
    assert decoded == ['text', b'\x00\xff', {'chunk': b'abc'}]


def test_legacy_primitive_pickle_is_read_without_unsafe_opt_in():
    legacy = ['chunk-1', {'usage': {'prompt_tokens': 12}}, b'raw']
    encoded = base64.b64encode(pickle.dumps(legacy)).decode('ascii')
    assert decode_data(encoded) == legacy


class _UnsafeLegacyPayload:
    def __reduce__(self):
        return (eval, ('1 + 1',))


def test_legacy_pickle_global_loading_is_blocked_by_default():
    import pytest

    encoded = base64.b64encode(pickle.dumps(_UnsafeLegacyPayload())).decode('ascii')
    with pytest.raises(ValueError, match='global/class loading'):
        decode_data(encoded)
