import asyncio
import numpy as np
import time
from typing import TYPE_CHECKING, AsyncIterator, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


async def _send_request(
    semaphore: asyncio.Semaphore,
    request: dict,
    is_warmup: bool,
    queue: asyncio.Queue,
    client: 'AioHttpClient',
    measure_client_gpu_memory: bool = False,
) -> None:
    async with semaphore:
        benchmark_data = await client.post(request)
    benchmark_data.is_warmup = is_warmup
    if measure_client_gpu_memory:
        benchmark_data.update_gpu_usage()
    await queue.put(benchmark_data)


class ClosedLoopStrategy(BenchmarkStrategy):
    """Closed-loop strategy with lazy request generation and bounded tasks."""

    def __init__(
        self,
        args: Arguments,
        api_plugin: 'ApiPluginBase',
        client: 'AioHttpClient',
        queue: asyncio.Queue,
        request_generator,
    ) -> None:
        super().__init__(args, api_plugin, client, queue)
        self._request_generator = request_generator

    async def run(self) -> None:
        request_iter = self._request_generator.__aiter__()
        warmup_count = self.args.warmup_count
        if warmup_count:
            await self._run_phase(request_iter, warmup_count, is_warmup=True, deadline=None)
        await self._run_phase(
            request_iter,
            int(self.args.number),
            is_warmup=False,
            deadline=self._compute_deadline(self.args.duration),
        )

    def _target_times(self, n: int) -> Optional[np.ndarray]:
        rate = self.args.rate
        if rate == -1 or n <= 0:
            return None
        intervals = np.random.exponential(1.0 / rate, size=n)
        delay_ts = np.cumsum(intervals)
        if self.args.arrival_distribution == 'normalized_poisson' and delay_ts[-1] > 0:
            delay_ts *= (n / rate) / delay_ts[-1]
        return delay_ts + time.perf_counter()

    async def _run_phase(
        self,
        request_iter: AsyncIterator[Tuple[dict, bool]],
        n: int,
        is_warmup: bool,
        deadline: Optional[float] = None,
    ) -> None:
        semaphore = asyncio.Semaphore(self.args.parallel)
        max_in_flight = self.args.parallel * self.args.in_flight_task_multiplier
        in_flight: set[asyncio.Task] = set()
        target_times = self._target_times(n)
        dispatched = 0

        for i in range(n):
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info(f'Duration deadline reached after dispatching {dispatched}/{n} requests.')
                break

            try:
                request, marked_warmup = await anext(request_iter)
            except StopAsyncIteration:
                break
            request_is_warmup = marked_warmup if marked_warmup == is_warmup else is_warmup

            if target_times is not None:
                sleep_s = target_times[i] - time.perf_counter()
                if deadline is not None:
                    sleep_s = min(sleep_s, deadline - time.perf_counter())
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            # A duration-limited phase must not dispatch one extra request just
            # because its scheduled sleep ended exactly at/after the deadline.
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info(f'Duration deadline reached after dispatching {dispatched}/{n} requests.')
                break

            if len(in_flight) >= max_in_flight:
                done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                await asyncio.gather(*done, return_exceptions=True)
                in_flight = set(pending)
            else:
                done = {task for task in in_flight if task.done()}
                if done:
                    await asyncio.gather(*done, return_exceptions=True)
                    in_flight.difference_update(done)

            task = asyncio.create_task(
                _send_request(
                    semaphore,
                    request,
                    request_is_warmup,
                    self.queue,
                    self.client,
                    self.args.measure_client_gpu_memory,
                )
            )
            in_flight.add(task)
            dispatched += 1

        if in_flight:
            if deadline is not None and time.perf_counter() >= deadline:
                logger.info(f'Duration deadline reached; awaiting {len(in_flight)} in-flight request(s).')
            await asyncio.gather(*in_flight, return_exceptions=True)
