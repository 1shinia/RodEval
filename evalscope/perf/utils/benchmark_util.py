from dataclasses import dataclass, field
import uuid
from typing import Any, List, Optional

from evalscope.perf.utils.perf_constants import Metrics
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

# ===========================================================================
# Layer 1: Single-request data container
# ===========================================================================


@dataclass
class BenchmarkData:
    """Data container for a single benchmark request/response cycle.

    Populated incrementally: raw fields are set by the HTTP client, then
    :meth:`finalize` is called once to derive timing/token metrics.
    """

    # --- Request ---
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Stable per-request identifier persisted in benchmark_data.db."""
    request: str = None  # JSON-serialized request body
    start_time: float = 0.0
    completed_time: float = 0.0
    chunk_times: List[float] = field(default_factory=list)
    success: bool = False
    response_messages: List[Any] = field(default_factory=list)

    # --- Derived timing (populated by finalize) ---
    query_latency: float = 0.0
    first_chunk_latency: float = 0.0
    """Time to the first protocol/data chunk. Diagnostic only; not TTFT."""
    first_token_latency: Optional[float] = None
    """Time to the first non-empty generated token/delta. This is the TTFT source."""
    last_generated_time: Optional[float] = None
    """Absolute perf-counter timestamp of the last generated-content chunk.

    Kept separate from ``completed_time`` so TPOT is not inflated by trailing
    usage-only SSE events, ``[DONE]`` markers, or protocol teardown.
    """
    time_per_output_token: Optional[float] = None
    """TPOT in seconds. Undefined for responses with fewer than two output tokens."""
    inter_chunk_latency: List[float] = field(default_factory=list)
    """Intervals between generated SSE chunks. This is ICL, not true token-level ITL."""
    max_gpu_memory_cost: float = 0.0
    """Optional *client-process* CUDA allocator peak, not remote server GPU memory."""

    # --- Response content ---
    generated_text: str = ''
    error: Optional[str] = None
    status_code: Optional[int] = None  # HTTP status code; set for non-200 responses
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # --- Multi-turn specific (only populated in multi-turn benchmark mode) ---
    input_num_turns: int = 0
    """Number of user turns in the conversation context when this request was sent."""
    real_cached_tokens: Optional[int] = None
    """Real cached token count from server response usage.prompt_tokens_details.cached_tokens."""
    cached_tokens: Optional[int] = None
    """Absolute number of KV-cached tokens for this turn.
    Populated only from server-reported cache telemetry.  Estimated reusable
    prefix tokens are kept separately in ``estimated_cacheable_tokens``."""
    estimated_cacheable_tokens: Optional[int] = None
    """Estimated reusable prefix length when the server does not report real cache hits.

    This is *not* a measured KV-cache hit and is therefore never folded into
    ``cached_tokens`` / the observed cache-hit-rate metric.
    """

    # --- Conversation-level progress signal (multi-turn only) ---
    is_last_turn: bool = False
    """True when this BenchmarkData is the final turn of its conversation.
    Used by the metrics consumer to advance the per-conversation progress bar."""

    is_first_turn: bool = False
    """True when this BenchmarkData is the first turn of its conversation.
    Used by per-turn-position metric grouping (cold vs. warm TTFT).
    Always False for single-turn benchmarks."""

    trace_id: Optional[str] = None
    """Stable identifier of the conversation this BenchmarkData belongs to,
    for trace-level aggregation in multi-turn benchmarks.  None for single-turn
    runs.  Format is opaque (e.g. ``trace-42`` from MultiTurnStrategy)."""

    # --- Warmup ---
    is_warmup: bool = False
    """True when this request is a warmup request, excluded from final metrics."""

    # --- Speculative decoding specific ---
    decoded_tokens_per_iter: Optional[float] = None
    """Average decoded tokens per decoder iteration from backend telemetry.

    Never inferred from SSE chunk counts because transport chunks are not
    decoder-iteration boundaries.
    """

    def finalize(self, api_plugin) -> None:
        """Parse token counts and compute all derived timing metrics.

        Must be called after the response is fully received.  Idempotent:
        token counts already present will not be re-parsed.
        """
        if self.prompt_tokens is None or self.completion_tokens is None:
            self.prompt_tokens, self.completion_tokens = api_plugin.parse_responses(
                self.response_messages, request=self.request
            )

        # A successful response with zero generated tokens has no TTFT.
        # Non-stream adapters initially set first_token_latency to E2E latency
        # because token timing is unobservable until parsing completes; clear
        # that synthetic value once token usage proves no token was generated.
        if self.completion_tokens is not None and self.completion_tokens <= 0:
            self.first_token_latency = None

        # TPOT = decode_span / (output_len - 1). A one-token response
        # has no inter-token decoding interval, so TPOT is intentionally None.
        if self.completion_tokens and self.completion_tokens > 1:
            # Non-stream responses set both first-token and first-chunk latency
            # to the total query latency (see default_api.py), so TTFT/TPOT are
            # not observable — report None (not-applicable) rather than 0.0.
            if (self.first_token_latency == self.query_latency
                    and self.first_chunk_latency == self.query_latency):
                self.time_per_output_token = None
            else:
                ttft = self.first_token_latency if self.first_token_latency is not None else self.first_chunk_latency
                decode_end_latency = self.query_latency
                if (
                    self.last_generated_time is not None
                    and self.start_time > 0
                    and self.last_generated_time >= self.start_time
                ):
                    decode_end_latency = self.last_generated_time - self.start_time
                self.time_per_output_token = max(
                    0.0, (decode_end_latency - ttft) / (self.completion_tokens - 1)
                )
        else:
            self.time_per_output_token = None

        # Derive inter-chunk latencies from chunk timestamps when not already set
        if not self.inter_chunk_latency and self.chunk_times:
            self.inter_chunk_latency = [t2 - t1 for t1, t2 in zip(self.chunk_times[:-1], self.chunk_times[1:])]

        # Do not infer speculative-decoding iterations from SSE chunk count.
        # SSE chunking is a transport/server-buffering property and is not a
        # decoder-iteration boundary. ``decoded_tokens_per_iter`` remains an
        # optional field for backends that can populate real server telemetry.

    def update_gpu_usage(self) -> None:
        """Update client-process CUDA allocator peak across visible devices.

        This does **not** represent memory used by a remote or separate-process
        inference server. The strategy layer keeps it opt-in to avoid perturbing
        high-throughput benchmark measurements.
        """
        if check_import('torch', raise_warning=False):
            import torch
            total_memory = sum(torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count()))
            self.max_gpu_memory_cost = max(self.max_gpu_memory_cost, total_memory)


# ===========================================================================
# Layer 2: Metric name constants + API type classification
# (Defined in perf_constants — single source of truth; imported above)
# ===========================================================================

# ===========================================================================
# Layer 3: Real-time metrics accumulator (mutable, updated per request)
# ===========================================================================


@dataclass
class MetricsAccumulator:
    """Stateful accumulator updated after every request during a benchmark run.

    Call :meth:`update` for each completed :class:`BenchmarkData`, then call
    :meth:`to_result` to obtain a computed :class:`BenchmarkMetrics` snapshot.
    """

    # --- Test configuration ---
    concurrency: int = 0
    rate: float = 0.0

    # --- Request counts ---
    n_total: int = 0
    n_success: int = 0

    # --- Wall-clock time window (private; exposed via wall_time property) ---
    _wall_start: Optional[float] = field(default=None, repr=False)
    _wall_end: Optional[float] = field(default=None, repr=False)

    # --- Cumulative sums (all use total_ prefix) ---
    total_latency: float = 0.0
    total_first_chunk_latency: float = 0.0
    total_first_token_latency: float = 0.0
    n_ttft_samples: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_time_per_output_token: float = 0.0
    n_tpot_samples: int = 0
    total_inter_chunk_latency: float = 0.0
    n_inter_chunk_samples: int = 0

    # --- Multi-turn cumulative sums ---
    total_input_turns: int = 0
    # Token-level accumulators for unbiased global cache hit rate.
    # total_cached_tokens: sum of cached_tokens across all turns (incl. turn 1 = 0).
    # total_prompt_tokens_for_cache: sum of prompt_tokens for turns that have
    # cached_tokens set (i.e. multi-turn turns).  Kept separate from
    # total_prompt_tokens so single-turn requests don't pollute the ratio.
    total_cached_tokens: int = 0
    total_prompt_tokens_for_cache: int = 0  # denominator: prompt_tokens of turns with cached_tokens set
    n_cache_turns: int = 0  # number of turns contributing to the cache ratio
    total_estimated_cacheable_tokens: int = 0
    total_prompt_tokens_for_estimated_cache: int = 0
    n_estimated_cache_turns: int = 0

    # First-turn vs subsequent-turn TTFT split (multi-turn only).
    # First-turn TTFT reflects cold prefill of the initial user prompt; subsequent
    # turns benefit from prefix-cache reuse and report markedly lower TTFT.
    # Only turns with input_num_turns > 0 (i.e. multi-turn turns) are bucketed,
    # so single-turn benchmarks leave both bucket counters at 0.
    total_first_turn_ttft: float = 0.0
    n_first_turn: int = 0
    total_subsequent_turn_ttft: float = 0.0
    n_subsequent_turn: int = 0

    # --- Speculative decoding cumulative sums ---
    total_decoded_tokens_per_iter: float = 0.0
    n_decoded_samples: int = 0

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def n_failed(self) -> int:
        """Number of failed requests (derived from totals)."""
        return self.n_total - self.n_success

    @property
    def wall_time(self) -> float:
        """Elapsed time from the earliest request start to the latest completion."""
        if self._wall_start is not None and self._wall_end is not None:
            return max(self._wall_end - self._wall_start, 0.0)
        return 0.0

    # -----------------------------------------------------------------------
    # Update
    # -----------------------------------------------------------------------

    def update(self, data: BenchmarkData, api_plugin) -> None:
        """Ingest one completed request and update all running totals."""
        if data.is_warmup:
            return  # Warmup requests are excluded from all metrics

        self.n_total += 1

        if data.success:
            self.n_success += 1
            data.finalize(api_plugin)

            self.total_latency += data.query_latency
            self.total_first_chunk_latency += data.first_chunk_latency
            if data.first_token_latency is not None:
                self.total_first_token_latency += data.first_token_latency
                self.n_ttft_samples += 1
            self.total_prompt_tokens += data.prompt_tokens or 0
            self.total_completion_tokens += data.completion_tokens or 0
            if data.time_per_output_token is not None:
                self.total_time_per_output_token += data.time_per_output_token
                self.n_tpot_samples += 1
            if data.inter_chunk_latency:
                self.total_inter_chunk_latency += sum(data.inter_chunk_latency)
                self.n_inter_chunk_samples += len(data.inter_chunk_latency)

            # Multi-turn specific
            if data.input_num_turns > 0:
                self.total_input_turns += data.input_num_turns
                # Bucket TTFT by turn position so cold prefill vs warm prefix-cache
                # turns can be reported separately. Single-turn runs (input_num_turns
                # == 0) are skipped to keep cold/warm distinction meaningful.
                if data.first_token_latency is not None:
                    if data.is_first_turn:
                        self.total_first_turn_ttft += data.first_token_latency
                        self.n_first_turn += 1
                    else:
                        self.total_subsequent_turn_ttft += data.first_token_latency
                        self.n_subsequent_turn += 1
            # Token-level cache accumulator: include *every* turn that has
            # cached_tokens set (turn 1 contributes 0 to numerator but its
            # prompt_tokens still count in the denominator).
            if data.cached_tokens is not None and data.prompt_tokens:
                self.total_cached_tokens += data.cached_tokens
                self.total_prompt_tokens_for_cache += data.prompt_tokens
                self.n_cache_turns += 1

            if data.estimated_cacheable_tokens is not None and data.prompt_tokens:
                self.total_estimated_cacheable_tokens += data.estimated_cacheable_tokens
                self.total_prompt_tokens_for_estimated_cache += data.prompt_tokens
                self.n_estimated_cache_turns += 1

            # Speculative-decoding metrics are accepted only when a backend
            # explicitly populated decoder telemetry.  Transport/SSE chunking
            # is deliberately not used as a proxy.
            if data.decoded_tokens_per_iter is not None and data.decoded_tokens_per_iter > 0:
                self.total_decoded_tokens_per_iter += data.decoded_tokens_per_iter
                self.n_decoded_samples += 1

        self._update_wall_time(data)

    def _update_wall_time(self, data: BenchmarkData) -> None:
        """Expand the wall-clock window to cover *data*'s lifecycle."""
        # A default-constructed failed BenchmarkData has start=completed=0.
        # Treat such records as missing timing rather than expanding the wall
        # window to the perf-counter epoch and collapsing throughput to ~0.
        if data.completed_time <= data.start_time:
            return
        if self._wall_start is None:
            self._wall_start = data.start_time
        else:
            self._wall_start = min(self._wall_start, data.start_time)

        if self._wall_end is None:
            self._wall_end = data.completed_time
        else:
            self._wall_end = max(self._wall_end, data.completed_time)

    # -----------------------------------------------------------------------
    # Result factory
    # -----------------------------------------------------------------------

    def to_result(self) -> 'BenchmarkMetrics':
        """Compute averages and return an immutable :class:`BenchmarkMetrics` snapshot."""
        n = self.n_success
        t = self.wall_time

        def _safe_div(numerator, denominator, default=-1):
            return numerator / denominator if denominator else default

        try:
            avg_latency = _safe_div(self.total_latency, n)
            avg_first_chunk_latency = _safe_div(self.total_first_chunk_latency, n)
            avg_first_token_latency = _safe_div(self.total_first_token_latency, self.n_ttft_samples, default=0.0)
            # Embedding/rerank APIs never produce output tokens, so no TTFT
            # sample is ever recorded; fall back to first-chunk (== full E2E
            # latency for one-shot APIs) so the summary stays readable instead
            # of collapsing to 0.0.  This also preserves the legacy report
            # value for the pathological all-zero-token LLM run (model
            # returning empty bodies); semantically-sound runs still use the
            # real first-token latency.
            if self.n_ttft_samples == 0 and n > 0:
                avg_first_token_latency = avg_first_chunk_latency
            avg_prompt_tokens = _safe_div(self.total_prompt_tokens, n)
            avg_completion_tokens = _safe_div(self.total_completion_tokens, n)
            avg_time_per_output_token = (
                _safe_div(self.total_time_per_output_token, self.n_tpot_samples, default=0.0)
                if self.n_tpot_samples else 0.0
            )
            avg_inter_chunk_latency = _safe_div(self.total_inter_chunk_latency, self.n_inter_chunk_samples, default=0.0)
            qps = _safe_div(n, t)
            # Workload-level input throughput must use elapsed wall time.  Using
            # sum(request_latency) undercounts concurrent embedding/rerank runs.
            avg_input_token_throughput = _safe_div(self.total_prompt_tokens, t)
            avg_output_token_throughput = _safe_div(self.total_completion_tokens, t)
            avg_total_token_throughput = _safe_div(self.total_prompt_tokens + self.total_completion_tokens, t)
            avg_turns_per_request = (_safe_div(self.total_input_turns, n) if self.total_input_turns > 0 else -1)
            # Unbiased global KV-cache hit rate:
            # total_cached_tokens / total_prompt_tokens_for_cache
            # (includes turn 1 which contributes 0 cached tokens but still
            # counts in the denominator, so the ratio is not inflated).
            avg_cached_percent = (
                _safe_div(self.total_cached_tokens
                          * 100.0, self.total_prompt_tokens_for_cache, default=-1) if self.n_cache_turns > 0 else -1
            )
            estimated_cacheable_percent = (
                _safe_div(
                    self.total_estimated_cacheable_tokens * 100.0,
                    self.total_prompt_tokens_for_estimated_cache,
                    default=-1,
                ) if self.n_estimated_cache_turns > 0 else -1
            )
            avg_decoded_tokens_per_iter = (
                _safe_div(self.total_decoded_tokens_per_iter, self.n_decoded_samples)
                if self.n_decoded_samples > 0 else -1
            )
            # First-turn / subsequent-turn TTFT averages (multi-turn only).
            # -1 means "not applicable" (no multi-turn data observed).
            avg_first_turn_ttft = (
                _safe_div(self.total_first_turn_ttft, self.n_first_turn) if self.n_first_turn > 0 else -1
            )
            avg_subsequent_turn_ttft = (
                _safe_div(self.total_subsequent_turn_ttft, self.n_subsequent_turn) if self.n_subsequent_turn > 0 else -1
            )
        except ZeroDivisionError as e:
            logger.error(
                f'ZeroDivisionError while computing metrics: {e}. '
                f'total_first_chunk_latency={self.total_first_chunk_latency}, '
                f'wall_time={t}, n_success={n}. '
                'This is likely caused by all requests returning empty responses. '
                'Please check the model service and ensure it is returning valid responses.'
            )
            avg_latency = avg_first_chunk_latency = avg_first_token_latency = avg_prompt_tokens = avg_completion_tokens = -1
            avg_time_per_output_token = avg_inter_chunk_latency = qps = -1
            avg_input_token_throughput = avg_output_token_throughput = avg_total_token_throughput = -1
            avg_turns_per_request = avg_cached_percent = estimated_cacheable_percent = avg_decoded_tokens_per_iter = -1
            avg_first_turn_ttft = avg_subsequent_turn_ttft = -1

        return BenchmarkMetrics(
            concurrency=self.concurrency,
            rate=self.rate,
            total_requests=self.n_total,
            succeed_requests=self.n_success,
            failed_requests=self.n_failed,
            total_time=t,
            avg_latency=avg_latency,
            avg_first_chunk_latency=avg_first_chunk_latency,
            avg_first_token_latency=avg_first_token_latency,
            avg_prompt_tokens=avg_prompt_tokens,
            avg_completion_tokens=avg_completion_tokens,
            avg_time_per_output_token=avg_time_per_output_token,
            avg_inter_chunk_latency=avg_inter_chunk_latency,
            qps=qps,
            avg_input_token_throughput=avg_input_token_throughput,
            avg_output_token_throughput=avg_output_token_throughput,
            avg_total_token_throughput=avg_total_token_throughput,
            avg_turns_per_request=avg_turns_per_request,
            avg_cached_percent=avg_cached_percent,
            estimated_cacheable_percent=estimated_cacheable_percent,
            avg_first_turn_ttft=avg_first_turn_ttft,
            avg_subsequent_turn_ttft=avg_subsequent_turn_ttft,
            avg_decoded_tokens_per_iter=avg_decoded_tokens_per_iter,
        )


# ===========================================================================
# Layer 4: Immutable result snapshot + serialization
# ===========================================================================


@dataclass
class BenchmarkMetrics:
    """Immutable snapshot of computed benchmark metrics.

    Produced by :meth:`MetricsAccumulator.to_result`.  Use
    :meth:`create_message` to serialize for logging or JSON export.
    """

    # --- Test configuration ---
    concurrency: int = 0
    rate: float = 0.0

    # --- Request statistics ---
    total_requests: int = 0
    succeed_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0

    # --- Latency averages ---
    avg_latency: float = -1
    avg_first_chunk_latency: float = -1
    avg_first_token_latency: float = -1
    avg_time_per_output_token: float = -1
    avg_inter_chunk_latency: float = -1

    # --- Throughput ---
    qps: float = -1
    avg_prompt_tokens: float = -1
    avg_completion_tokens: float = -1
    avg_input_token_throughput: float = -1
    avg_output_token_throughput: float = -1
    avg_total_token_throughput: float = -1

    # --- Multi-turn ---
    avg_turns_per_request: float = -1
    avg_cached_percent: float = -1
    estimated_cacheable_percent: float = -1
    avg_first_turn_ttft: float = -1
    """Avg TTFT (seconds) of first-turn requests (cold prefill).  -1 = not applicable."""
    avg_subsequent_turn_ttft: float = -1
    """Avg TTFT (seconds) of subsequent-turn requests (warm prefix cache).  -1 = not applicable."""

    # --- Speculative decoding ---
    avg_decoded_tokens_per_iter: float = -1
    """Average decoded tokens per decoder iteration from backend telemetry."""

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def create_message(self, ndigits: int = 4, api_type: str = None) -> dict:
        """Build a metrics dictionary suitable for logging or JSON export.

        Args:
            ndigits: Decimal places for rounding float values.
            api_type: API name string; selects LLM vs Embedding field set.
        """
        base = self._build_common_fields(ndigits)
        specific = (
            self._build_embedding_fields(ndigits)
            if Metrics.is_embedding_or_rerank(api_type) else self._build_llm_fields(ndigits)
        )
        multiturn = self._build_multiturn_fields(ndigits)
        speculative = self._build_speculative_decoding_fields(ndigits)
        return {**base, **specific, **multiturn, **speculative}

    def _build_common_fields(self, r: int) -> dict:
        """Fields shared by all API types."""
        return {
            Metrics.TIME_TAKEN_FOR_TESTS: round(self.total_time, r),
            Metrics.NUMBER_OF_CONCURRENCY: self.concurrency,
            Metrics.REQUEST_RATE: self.rate,
            Metrics.TOTAL_REQUESTS: int(self.total_requests),
            Metrics.SUCCEED_REQUESTS: self.succeed_requests,
            Metrics.FAILED_REQUESTS: self.failed_requests,
            Metrics.REQUEST_THROUGHPUT: round(self.qps, r),
            Metrics.AVERAGE_LATENCY: round(self.avg_latency, r),
            Metrics.AVERAGE_INPUT_TOKENS_PER_REQUEST: round(self.avg_prompt_tokens, r),
        }

    def _build_llm_fields(self, r: int) -> dict:
        """Additional fields for LLM text-generation APIs."""
        return {
            Metrics.OUTPUT_TOKEN_THROUGHPUT: round(self.avg_output_token_throughput, r),
            Metrics.TOTAL_TOKEN_THROUGHPUT: round(self.avg_total_token_throughput, r),
            Metrics.AVERAGE_TIME_TO_FIRST_TOKEN: round(self.avg_first_token_latency * 1000, 2),
            Metrics.AVERAGE_TIME_TO_FIRST_CHUNK: round(self.avg_first_chunk_latency * 1000, 2),
            Metrics.AVERAGE_TIME_PER_OUTPUT_TOKEN: round(self.avg_time_per_output_token * 1000, 2),
            Metrics.AVERAGE_INTER_CHUNK_LATENCY: round(self.avg_inter_chunk_latency * 1000, 2),
            Metrics.AVERAGE_OUTPUT_TOKENS_PER_REQUEST: round(self.avg_completion_tokens, r),
        }

    def _build_embedding_fields(self, r: int) -> dict:
        """Additional fields for Embedding / Rerank APIs."""
        return {
            Metrics.INPUT_TOKEN_THROUGHPUT: round(self.avg_input_token_throughput, r),
        }

    def _build_multiturn_fields(self, r: int) -> dict:
        """Conditionally included multi-turn conversation metrics."""
        result = {}
        if self.avg_turns_per_request > 0:
            result[Metrics.AVERAGE_INPUT_TURNS_PER_REQUEST] = round(self.avg_turns_per_request, r)
        # Emit whenever multi-turn cache tracking was active (avg_cached_percent >= 0).
        # -1 means "not applicable" (no multi-turn data); 0 means active but no cache hits.
        if self.avg_cached_percent >= 0:
            result[Metrics.AVERAGE_CACHED_PERCENT] = round(self.avg_cached_percent, r)
        if self.estimated_cacheable_percent >= 0:
            result[Metrics.ESTIMATED_REUSABLE_PREFIX_PERCENT] = round(self.estimated_cacheable_percent, r)
        # First-turn / subsequent-turn TTFT split (multi-turn only).  Stored in
        # seconds; report in ms for consistency with avg_ttft.
        if self.avg_first_turn_ttft >= 0:
            result[Metrics.AVERAGE_FIRST_TURN_TTFT] = round(self.avg_first_turn_ttft * 1000, 2)
        if self.avg_subsequent_turn_ttft >= 0:
            result[Metrics.AVERAGE_SUBSEQUENT_TURN_TTFT] = round(self.avg_subsequent_turn_ttft * 1000, 2)
        return result

    def _build_speculative_decoding_fields(self, r: int) -> dict:
        """Conditionally included speculative decoding metrics.

        Only emitted when a backend populated real decoder-iteration telemetry.

        - ``avg_decoded_tokens_per_iter`` (L): average accepted tokens per
          speculative-decoding iteration, supplied by backend telemetry.

        Acceptance rate is intentionally *not* inferred from L: converting
        tokens/iteration into a draft-token acceptance probability requires
        backend-specific information such as draft length and iteration
        semantics.  A future backend may expose acceptance rate directly.
        """
        result = {}
        if self.avg_decoded_tokens_per_iter > 0:
            L = self.avg_decoded_tokens_per_iter
            result[Metrics.AVERAGE_DECODED_TOKENS_PER_ITER] = round(L, r)
        return result
