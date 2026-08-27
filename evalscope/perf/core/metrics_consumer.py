import asyncio
import json
import sqlite3
import time
from tqdm import tqdm as tqdm_std
from typing import TYPE_CHECKING, Tuple

from evalscope.constants import HEARTBEAT_INTERVAL_SEC
from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient, test_connection
from evalscope.perf.utils.benchmark_util import Metrics, MetricsAccumulator
from evalscope.perf.utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data_batch
from evalscope.perf.utils.handler import exception_handler
from evalscope.perf.utils.log_utils import maybe_log_to_visualizer
from evalscope.perf.utils.trace_metrics import TraceAccumulator, TraceLevelSummary
from evalscope.perf.utils.workload_timeline import WorkloadTimeline
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

if TYPE_CHECKING:
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()

# Global event signalling that all requests have been dispatched and the
# metrics consumer should flush remaining items and exit.
data_process_completed_event = asyncio.Event()


@exception_handler
async def statistic_benchmark_metric(
    benchmark_data_queue: asyncio.Queue,
    args: Arguments,
    api_plugin: 'ApiPluginBase',
) -> Tuple['MetricsAccumulator', 'TraceLevelSummary', 'WorkloadTimeline', str]:
    """Consume benchmark results from the queue, update metrics, and persist to DB.

    Args:
        benchmark_data_queue: Queue populated by request workers.
        args: Benchmark configuration.
        api_plugin: API plugin used to finalise token counts.

    Returns:
        Tuple of ``(metrics_accumulator_result, trace_level_summary, workload_timeline, result_db_path)``.
        ``trace_level_summary`` is empty for single-turn runs (no ``trace_id``);
        ``workload_timeline`` always accumulates regardless of mode, callers
        may inspect ``n_points`` before rendering downstream tables.
    """
    accumulator = MetricsAccumulator(concurrency=args.parallel, rate=args.rate)
    trace_acc = TraceAccumulator()
    workload_timeline = WorkloadTimeline()
    result_db_path = get_result_db_path(args)
    warmup_count = args.warmup_count

    # Stream to a dedicated DB writer so request-result consumption is not
    # coupled to SQLite serialization/fsync latency.
    commit_every = args.db_commit_interval

    with sqlite3.connect(result_db_path, check_same_thread=False) as con:
        cursor = con.cursor()
        create_result_table(cursor)
        con.commit()

        parallel_for_queue = max(args.parallel) if isinstance(args.parallel, list) else args.parallel
        db_queue_maxsize = max(commit_every * 4, max(1, parallel_for_queue) * args.queue_size_multiplier * 2)
        db_write_queue: asyncio.Queue = asyncio.Queue(maxsize=db_queue_maxsize)
        last_db_queue_warning = 0.0

        def _flush_db_batch(batch: list) -> None:
            batch_cursor = con.cursor()
            insert_benchmark_data_batch(
                batch_cursor,
                batch,
                store_payloads=args.db_store_payloads,
            )
            con.commit()

        async def _db_writer() -> None:
            pending = []
            flush_interval_sec = 1.0
            pending_since = None

            async def _flush_pending(*, final: bool = False) -> None:
                nonlocal pending, pending_since
                if not pending:
                    return
                started = time.monotonic()
                batch = pending
                pending = []
                await asyncio.to_thread(_flush_db_batch, batch)
                pending_since = None
                elapsed = time.monotonic() - started
                if elapsed > 1.0:
                    kind = 'final flush' if final else 'flush'
                    logger.warning(f'Benchmark DB {kind} took {elapsed:.2f}s for {len(batch)} rows')

            while True:
                # Keep row-count batching for throughput, but bound how long
                # completed requests can live only in memory during low-rate
                # or short-running benchmarks.
                timeout = None
                if pending:
                    timeout = max(0.0, flush_interval_sec - (time.monotonic() - pending_since))
                try:
                    if timeout is None:
                        item = await db_write_queue.get()
                    else:
                        item = await asyncio.wait_for(db_write_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    await _flush_pending()
                    continue

                try:
                    if item is None:
                        await _flush_pending(final=True)
                        return

                    if not pending:
                        pending_since = time.monotonic()
                    pending.append(item)
                    if len(pending) >= commit_every:
                        await _flush_pending()
                finally:
                    db_write_queue.task_done()

        db_writer_task = asyncio.create_task(_db_writer())

        writer_shutdown_sent = False
        try:
            cur_run_name = (
                f'rate_{args.rate}_number_{args.number}'
                if args.open_loop else f'parallel_{args.parallel}_number_{args.number}'
            )

            # Warmup bar
            _warmup_pbar = None
            if warmup_count > 0:
                _warmup_pbar = tqdm_std(
                    desc=f'Warmup[{cur_run_name}]',
                    total=warmup_count,
                )

            with tqdm(
                desc=f'Processing[{cur_run_name}]',
                total=args.number,
                logger=logger,
                log_interval=HEARTBEAT_INTERVAL_SEC,
                track_progress=True,
            ) as pbar:
                while not (data_process_completed_event.is_set() and benchmark_data_queue.empty()):
                    try:
                        benchmark_data = await asyncio.wait_for(benchmark_data_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if benchmark_data.is_warmup:
                        benchmark_data_queue.task_done()
                        # Multi-turn: only count last turn per conversation.
                        if not benchmark_data.is_last_turn and benchmark_data.input_num_turns > 0:
                            continue
                        if _warmup_pbar:
                            _warmup_pbar.update(1)
                        continue

                    # First benchmark item — close the warmup bar so it disappears.
                    if _warmup_pbar:
                        _warmup_pbar.close()
                        _warmup_pbar = None

                    # Update metrics immediately; DB persistence is delegated to
                    # the bounded writer queue and flushed with executemany.
                    accumulator.update(benchmark_data, api_plugin)
                    # Feed the per-trace accumulator *after* MetricsAccumulator.update
                    # so finalize() has populated prompt/completion tokens (idempotent).
                    # Single-turn items (trace_id is None) are silently skipped inside.
                    trace_acc.feed(benchmark_data)
                    # Workload timeline tracks cumulative tokens vs time for the
                    # Overall / Last-window / Steady-state throughput breakdown.
                    workload_timeline.feed(benchmark_data)
                    if db_writer_task.done():
                        await db_writer_task
                    await db_write_queue.put(benchmark_data)
                    now = time.monotonic()
                    if (
                        db_write_queue.qsize() >= int(db_queue_maxsize * 0.75)
                        and now - last_db_queue_warning >= 5.0
                    ):
                        logger.warning(
                            f'Benchmark DB writer queue is {db_write_queue.qsize()}/{db_queue_maxsize}; '
                            'SQLite persistence is approaching benchmark backpressure'
                        )
                        last_db_queue_warning = now

                    message = accumulator.to_result().create_message(api_type=args.api)

                    await asyncio.to_thread(maybe_log_to_visualizer, args, message)

                    if int(accumulator.n_total) % args.log_every_n_query == 0:
                        msg = json.dumps(message, ensure_ascii=False, indent=2)
                        logger.info(msg)

                    benchmark_data_queue.task_done()
                    # In multi-turn mode each conversation produces multiple turns;
                    # advance the progress bar only once per conversation (on the
                    # last turn).  In single-turn mode is_last_turn is always False,
                    # so we fall back to updating on every item.
                    if not benchmark_data.is_last_turn and benchmark_data.input_num_turns > 0:
                        continue
                    pbar.update(1)

            await db_write_queue.put(None)
            writer_shutdown_sent = True
            await db_writer_task
        finally:
            if not db_writer_task.done():
                if not writer_shutdown_sent:
                    try:
                        db_write_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        db_writer_task.cancel()
                try:
                    await db_writer_task
                except asyncio.CancelledError:
                    pass
                except Exception as writer_error:
                    logger.error(f'Benchmark DB writer failed during cleanup: {writer_error}')

    return accumulator.to_result(), trace_acc.to_summary(), workload_timeline, result_db_path


@exception_handler
async def connect_test(args: Arguments, api_plugin: 'ApiPluginBase') -> None:
    """Perform a connection test unless disabled or not applicable.

    Raises:
        TimeoutError: If the test connection fails.
    """
    if Metrics.is_embedding_or_rerank(args.api):
        return

    if args.no_test_connection:
        return

    if not await test_connection(args, api_plugin):
        raise TimeoutError('Test connection failed')
