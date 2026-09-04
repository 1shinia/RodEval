import asyncio
import numpy as np
from typing import TYPE_CHECKING, AsyncGenerator, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.core.metrics_consumer import connect_test, data_process_completed_event, statistic_benchmark_metric
from evalscope.perf.core.strategies import ClosedLoopStrategy, OpenLoopStrategy
from evalscope.perf.plugin import ApiRegistry, DatasetRegistry
from evalscope.perf.utils.db_util import load_prompt, summary_result
from evalscope.perf.utils.handler import exception_handler
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

if TYPE_CHECKING:
    from evalscope.perf.plugin import ApiPluginBase
    from evalscope.perf.utils.perf_models import BenchmarkSummary, PercentileResult
    from evalscope.perf.utils.trace_metrics import TraceLevelSummary
    from evalscope.perf.utils.workload_timeline import WorkloadThroughput

logger = get_logger()


@exception_handler
async def get_requests(args: Arguments, api_plugin: 'ApiPluginBase') -> AsyncGenerator[Tuple[dict, bool], None]:
    """Generate requests with warmup marking.

    Yields ``(request_dict, is_warmup)`` tuples.  The first ``warmup_count``
    requests are marked ``is_warmup=True`` and excluded from final metrics.
    Total yield count = ``warmup_count + args.number``.
    """
    warmup_count = args.warmup_count
    total_count = args.total_count

    if warmup_count > 0:
        logger.info(
            f'Warmup enabled: {warmup_count} warmup requests '
            f'(total: {total_count}, benchmark: {args.number})'
        )

    async def _generate_from_prompt():
        """Generate requests by repeating a single prompt."""
        prompt = load_prompt(args.prompt)
        messages = [{'role': 'user', 'content': prompt}] if args.apply_chat_template else prompt
        request = api_plugin.build_request(messages)
        for i in range(total_count):
            yield request, i < warmup_count

    async def _generate_from_dataset():
        """Generate requests without materialising the requested run in memory.

        When the requested count exceeds the dataset length we re-open the
        dataset iterator for another pass instead of retaining every message.
        """
        count = 0
        dataset_cls = DatasetRegistry.get_class(args.dataset)
        with tqdm(desc='Generating[requests]', total=total_count, logger=logger) as pbar:
            while count < total_count:
                # Recreate the adapter each pass so one-shot dataset iterators
                # remain cyclable without retaining the whole dataset in RAM.
                message_generator = dataset_cls(args)
                produced_this_pass = 0
                for messages in message_generator.build_messages():
                    request = api_plugin.build_request(messages)
                    if request is None:
                        continue
                    yield request, count < warmup_count
                    count += 1
                    produced_this_pass += 1
                    pbar.update(1)
                    if count >= total_count:
                        break
                if produced_this_pass == 0:
                    raise ValueError('Dataset is empty or produced no valid requests!')

    # Dispatch based on arguments.
    if args.prompt:
        generator = _generate_from_prompt()
    elif args.dataset:
        generator = _generate_from_dataset()
    else:
        raise ValueError('Either prompt or dataset is required!')

    # Yield requests without rate limiting; open-loop strategies apply the
    # Poisson sleep in their dispatch loop so the interval falls between
    # consecutive dispatches rather than during pre-collection.
    async for request, is_warmup in generator:
        yield request, is_warmup


@exception_handler
async def run_benchmark(
    args: Arguments,
) -> Tuple['BenchmarkSummary', 'PercentileResult', Optional['TraceLevelSummary'], Optional['WorkloadThroughput']]:
    """Run a single-turn benchmark.

    Dispatches requests using either :class:`~evalscope.perf.core.strategies.OpenLoopStrategy`
    or :class:`~evalscope.perf.core.strategies.ClosedLoopStrategy` depending on
    ``args.open_loop``.

    Args:
        args: Benchmark configuration.

    Returns:
        4-tuple of ``(summary, percentiles, trace_summary, workload_throughput)``.
    """
    api_plugin_class = ApiRegistry.get_class(args.api)
    api_plugin = api_plugin_class(args)

    await connect_test(args, api_plugin)

    if args.open_loop:
        queue: asyncio.Queue = asyncio.Queue(maxsize=args.open_loop_result_queue_maxsize)
    else:
        queue = asyncio.Queue(maxsize=max(1, args.parallel * args.queue_size_multiplier))

    data_process_completed_event.clear()

    client = AioHttpClient(args, api_plugin)
    async with client:
        statistic_task = asyncio.create_task(statistic_benchmark_metric(queue, args, api_plugin))

        try:
            request_gen = get_requests(args, api_plugin)
            if args.open_loop:
                strategy = OpenLoopStrategy(args, api_plugin, client, queue, request_gen)
            else:
                strategy = ClosedLoopStrategy(args, api_plugin, client, queue, request_gen)

            await strategy.run()

            await queue.join()
        finally:
            # Always signal the consumer loop to stop, even if strategy.run()
            # or queue.join() raised an exception.  Without this, the consumer
            # loop spins indefinitely waiting for the queue to drain.
            data_process_completed_event.set()

        metrics, trace_summary, workload_timeline, result_db_path = await statistic_task

    return summary_result(
        args,
        metrics,
        result_db_path,
        trace_summary=trace_summary,
        workload_timeline=workload_timeline,
    )
