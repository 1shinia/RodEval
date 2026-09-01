"""Cumulative token timeline + workload-level throughput aggregator.

Mirrors trie's ``CompletionPoint`` / ``ServerMetrics`` reporting so multi-turn
runs can report throughput under three windows that single-request averages
would smear together:

* ``Overall``        - total tokens / total wall-clock time
* ``Last 30s``       - sliding tail-window, captures end-of-run behaviour
* ``Steady-state``   - drops the first ``warmup_frac`` (default 20%) of
  wall-clock to exclude the ramp where the server is still spinning up its
  KV cache, request scheduler, and (for batched engines) reaching steady
  batch sizes.  Matches trie's steady-state definition.

Timeline points are kept in memory only (raw points may also be exported
to ``workload_timeline.json`` so downstream tools / notebooks can re-derive
custom windows with pandas).  The SQLite layer is intentionally untouched.

Per-point fields:

* ``t``                  - seconds since wall-clock start (earliest valid
  request ``start_time``)
* ``cum_completion``     - cumulative completion tokens
* ``cum_new_prompt``     - cumulative ``prompt_tokens - cached_tokens``
  where ``cached_tokens`` means *server-observed* cache hits only
* ``cum_cached_prompt``  - cumulative server-observed ``cached_tokens``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from tabulate import tabulate
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from evalscope.perf.utils.benchmark_util import BenchmarkData

# ---------------------------------------------------------------------------
# Internal point
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    t: float
    cum_completion: int
    cum_new_prompt: int
    cum_cached_prompt: int


@dataclass
class _Record:
    start_time: float
    completed_time: float
    success: bool
    completion: int
    new_prompt: int
    cached_prompt: int


# Zero-anchor used for the "Overall" window: makes overall_rates share the
# same code path as the windowed rates without a special-case branch.
_ORIGIN = _Point(t=0.0, cum_completion=0, cum_new_prompt=0, cum_cached_prompt=0)

# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class WorkloadTimeline:
    """Completion timeline rebuilt from absolute request timestamps.

    Feed every non-warmup :class:`BenchmarkData` via :meth:`feed`. Failed
    attempts contribute zero tokens but still extend the wall-clock window;
    otherwise a slow final failure would incorrectly increase reported
    goodput. Call :meth:`to_summary` at the end of the run to obtain a
    :class:`WorkloadThroughput` snapshot.

    Requests are stored as absolute records and sorted by completion time when
    metrics are read.  This is important under concurrency: the first request
    to *finish* is not necessarily the first request that *started*.
    """

    _wall_start: Optional[float] = None
    _points: List[_Point] = field(default_factory=list)
    _records: List[_Record] = field(default_factory=list, repr=False)
    _dirty: bool = field(default=False, repr=False)

    def feed(self, data: 'BenchmarkData') -> None:
        if data.is_warmup:
            return
        # Default-constructed failures have 0/0 timestamps.  Successful records
        # must have a positive-width lifecycle.
        if data.completed_time <= data.start_time:
            return

        prompt = (data.prompt_tokens or 0) if data.success else 0
        completion = (data.completion_tokens or 0) if data.success else 0
        cached = (data.cached_tokens or 0) if data.success else 0
        # Server-side new-prompt cost.  Clamp to >=0 because some servers can
        # report cached > prompt when the chat template inflates the prompt
        # post-tokenization (rare but observed on a few OpenAI-compat backends).
        new_prompt = max(prompt - cached, 0)

        self._records.append(
            _Record(
                start_time=data.start_time,
                completed_time=data.completed_time,
                success=data.success,
                completion=completion,
                new_prompt=new_prompt,
                cached_prompt=cached,
            )
        )
        self._dirty = True

    def _ensure_points(self) -> None:
        """Rebuild cumulative points from absolute records when necessary."""
        if not self._dirty:
            return
        if not self._records:
            self._wall_start = None
            self._points = []
            self._dirty = False
            return

        self._wall_start = min(r.start_time for r in self._records)
        records = sorted(self._records, key=lambda r: r.completed_time)

        cum_completion = 0
        cum_new_prompt = 0
        cum_cached_prompt = 0
        points: List[_Point] = []
        for record in records:
            cum_completion += record.completion
            cum_new_prompt += record.new_prompt
            cum_cached_prompt += record.cached_prompt
            points.append(
                _Point(
                    t=record.completed_time - self._wall_start,
                    cum_completion=cum_completion,
                    cum_new_prompt=cum_new_prompt,
                    cum_cached_prompt=cum_cached_prompt,
                )
            )
        self._points = points
        self._dirty = False

    # ------------------------------------------------------------------
    # Derived rates
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return sum(1 for r in self._records if r.success)

    @property
    def wall_time(self) -> float:
        """End-to-end wall-clock duration (first start -> last completion)."""
        self._ensure_points()
        if not self._points:
            return 0.0
        return max(self._points[-1].t, 0.0)

    def _rates_from(self, anchor: _Point) -> List[float]:
        """Return [total_prompt, new_prompt, cached_prompt, completion] tok/s
        for the window from ``anchor`` to the latest point.

        Returns zeros when the window has no width or no points.
        """
        self._ensure_points()
        if not self._points:
            return [0.0, 0.0, 0.0, 0.0]
        last = self._points[-1]
        dt = last.t - anchor.t
        if dt <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        d_new_prompt = last.cum_new_prompt - anchor.cum_new_prompt
        d_cached_prompt = last.cum_cached_prompt - anchor.cum_cached_prompt
        d_completion = last.cum_completion - anchor.cum_completion
        return [
            (d_new_prompt + d_cached_prompt) / dt,
            d_new_prompt / dt,
            d_cached_prompt / dt,
            d_completion / dt,
        ]

    def overall_rates(self) -> List[float]:
        return self._rates_from(_ORIGIN)

    def last_window_rates(self, window_s: float) -> List[float]:
        """Sliding tail window of length ``window_s`` seconds.

        If the run is shorter than ``window_s`` we fall back to ``overall_rates``
        - quoting trie's "Last 30s" on a 5s run would otherwise be misleading.
        """
        self._ensure_points()
        if not self._points or window_s <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        wall = self.wall_time
        if wall <= window_s:
            return self.overall_rates()
        # Use an exact time-boundary anchor.  Cumulative completions form a step
        # function, so the token count at an arbitrary timestamp is well-defined
        # even when no request completed exactly at that instant.
        return self._rates_from(self._find_anchor(wall - window_s))

    def steady_state_rates(self, warmup_frac: float = 0.2) -> List[float]:
        """Drop the first ``warmup_frac`` of wall_time and rate over what remains.

        Matches trie's steady-state window: takes the latest cumulative counts
        and subtracts the counts at the moment ``warmup_frac * wall_time`` elapsed.
        Falls back to ``overall_rates`` when the timeline is too short to
        meaningfully discard a warmup region.
        """
        self._ensure_points()
        if not self._points:
            return [0.0, 0.0, 0.0, 0.0]
        wall = self.wall_time
        if wall <= 0 or warmup_frac <= 0:
            return self.overall_rates()
        if warmup_frac >= 1:
            return self.overall_rates()
        # A single completion point cannot carve a meaningful steady-state
        # window; fall back to overall rates rather than fabricating a rate
        # over a window that contains no additional sample.
        if len(self._points) < 2:
            return self.overall_rates()
        # If no completion landed inside the warmup window, dropping the first
        # warmup_frac of wall time removes no output and only shrinks the
        # denominator — inflating the steady-state rate. Fall back to overall
        # rather than reporting a fabricated rate.
        if self._points[0].t > wall * warmup_frac:
            return self.overall_rates()
        anchor = self._find_anchor(wall * warmup_frac)
        return self._rates_from(anchor)

    def _find_anchor(self, t_target: float) -> _Point:
        """Return cumulative counts at the exact timestamp ``t_target``."""
        self._ensure_points()
        if not self._points:
            return _Point(t=max(t_target, 0.0), cum_completion=0, cum_new_prompt=0, cum_cached_prompt=0)

        chosen = _ORIGIN
        for p in self._points:
            if p.t <= t_target:
                chosen = p
            else:
                break
        return _Point(
            t=max(t_target, 0.0),
            cum_completion=chosen.cum_completion,
            cum_new_prompt=chosen.cum_new_prompt,
            cum_cached_prompt=chosen.cum_cached_prompt,
        )

    # ------------------------------------------------------------------
    # Snapshot / serialisation
    # ------------------------------------------------------------------

    def to_summary(self, *, last_window_s: float = 30.0, warmup_frac: float = 0.2) -> 'WorkloadThroughput':
        overall = self.overall_rates()
        last = self.last_window_rates(last_window_s)
        steady = self.steady_state_rates(warmup_frac)
        labels = [
            'Total Prompt tok/s',
            'New Prompt tok/s',
            'Cached Prompt tok/s',
            'Completion tok/s',
        ]
        rows = [
            WorkloadThroughputRow(metric=label, overall=overall[i], last_window=last[i], steady_state=steady[i])
            for i, label in enumerate(labels)
        ]
        return WorkloadThroughput(
            n_samples=self.n_points,
            wall_time_s=round(self.wall_time, 4),
            last_window_s=last_window_s,
            warmup_frac=warmup_frac,
            rows=rows,
        )

    def to_raw_points_dict(self) -> dict:
        """Export raw cumulative-token points for downstream pandas / plots."""
        self._ensure_points()
        return {
            'wall_start': self._wall_start,
            'points': [{
                't': round(p.t, 6),
                'cum_completion': p.cum_completion,
                'cum_new_prompt': p.cum_new_prompt,
                'cum_cached_prompt': p.cum_cached_prompt,
            } for p in self._points],
        }


# ---------------------------------------------------------------------------
# Pydantic snapshot
# ---------------------------------------------------------------------------


class WorkloadThroughputRow(BaseModel):
    """One throughput row across Overall / Last-window / Steady-state columns."""

    metric: str
    overall: float = 0.0
    last_window: float = 0.0
    steady_state: float = 0.0


class WorkloadThroughput(BaseModel):
    """Workload-level throughput summary table.

    ``last_window_s`` and ``warmup_frac`` are recorded alongside the values so a
    consumer can interpret the columns without guessing what window was used.
    """

    n_samples: int = 0
    wall_time_s: float = 0.0
    last_window_s: float = 30.0
    warmup_frac: float = 0.2
    rows: List[WorkloadThroughputRow] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.rows

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_table(self) -> str:
        if self.is_empty():
            return ''
        last_label = f'Last {int(self.last_window_s)}s'
        steady_label = f'Steady (drop {int(self.warmup_frac * 100)}%)'
        headers = ['Metric (tok/s)', 'Overall', last_label, steady_label]
        body = [[
            r.metric,
            f'{r.overall:.2f}',
            f'{r.last_window:.2f}',
            f'{r.steady_state:.2f}',
        ] for r in self.rows]
        col_align = ('left', ) + ('right', ) * 3
        return tabulate(
            body,
            headers=headers,
            tablefmt='simple_outline',
            disable_numparse=True,
            colalign=col_align,
        )
