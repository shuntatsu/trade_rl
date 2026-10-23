"""Stream one arm's parsed event rows against its supplied static binding."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from tools.market_replay_oracle.event_rows import actor_events, reserve_events
from tools.market_replay_oracle.events import OrderEventContext, OrderEventEvidence
from tools.market_replay_oracle.exact_fill_binding import ExactFillAttemptBinding

_ACTOR_ROWS = 6527
_RESERVE_ROWS = 72
_END = object()


@dataclass(frozen=True, slots=True)
class BoundEventAdvance:
    """One checked row projection, including any original native rejection flag."""

    count: int
    phase: Literal["actor", "reserve"]
    context: OrderEventContext
    events: tuple[OrderEventEvidence, ...] | None
    native_rejected: bool


@dataclass(frozen=True, slots=True)
class BoundEventRowCoverage:
    """Presence/consistency counts; never an authenticated attempt or ledger."""

    binding_digest: str
    account: str
    scenario: str
    role: str
    expected_actor_rows: int
    expected_reserve_rows: int
    observed_actor_rows: int
    observed_reserve_rows: int
    parsed_event_batches: int
    event_count: int
    rejected_native_rows: int
    tail_native_missing: bool

    @property
    def scope(self) -> str:
        return "BOUND_EVENT_ROW_COVERAGE_ONLY"

    @property
    def all_expected_batches_present(self) -> bool:
        """Batch presence only; a rejected row can still have a present batch."""
        return (
            self.observed_actor_rows == self.expected_actor_rows
            and self.observed_reserve_rows == self.expected_reserve_rows
            and self.parsed_event_batches
            == self.expected_actor_rows + self.expected_reserve_rows
            and not self.tail_native_missing
        )


def _rows(values: Iterable[object]) -> Iterator[object]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise ValueError("event rows must be row iterables, not a scalar or object")
    try:
        return iter(values)
    except TypeError as error:
        raise ValueError("event rows must be iterable") from error


def _bound_clock(binding: ExactFillAttemptBinding) -> None:
    if binding.payload()["clock"] != {
        "actor_start": 2903,
        "actor_stop": 9430,
        "settlement_stop": 9502,
    }:
        raise ValueError("event rows require the frozen actor/reserve clock")


def _advances(
    binding: ExactFillAttemptBinding,
    arm: tuple[str, str, str],
    actor: Iterator[object],
    reserve: Iterator[object],
) -> Iterator[BoundEventAdvance]:
    actor_count = 0
    for actor_count, row in enumerate(actor, 1):
        if actor_count > _ACTOR_ROWS:
            raise ValueError("actor event rows exceed the frozen horizon")
        context = binding.event_context(*arm, actor_count)
        events, rejected = actor_events(row, context)
        if events is None:
            if next(actor, _END) is not _END or next(reserve, _END) is not _END:
                raise ValueError("missing native facts are only valid at an actor tail")
            yield BoundEventAdvance(actor_count, "actor", context, None, rejected)
            return
        yield BoundEventAdvance(actor_count, "actor", context, events, rejected)
    for offset, row in enumerate(reserve, 1):
        if actor_count != _ACTOR_ROWS or offset > _RESERVE_ROWS:
            raise ValueError(
                "reserve rows require a full actor prefix and bounded horizon"
            )
        count = _ACTOR_ROWS + offset
        context = binding.event_context(*arm, count)
        events, rejected = reserve_events(row, context)
        yield BoundEventAdvance(count, "reserve", context, events, rejected)


def walk_bound_event_rows(
    binding: ExactFillAttemptBinding,
    *,
    account: str,
    scenario: str,
    role: str,
    actor_rows: Iterable[object],
    closing_rows: Iterable[object],
    on_advance: Callable[[BoundEventAdvance], None] | None = None,
) -> BoundEventRowCoverage:
    """Validate/stream row projections; no outer bytes, receipt or attempt authority.

    Rows may be prefixes. Missing keys never mean empty batches; only a final
    explicit actor ``native: null`` retains a missing batch. Rejection flags do
    not disappear when events parse. Child JSON is derived, not source bytes.
    Iterator/consumer errors propagate without returning a coverage summary.
    The consumer owns any retained output; this walker keeps only one batch.
    """
    if on_advance is not None and not callable(on_advance):
        raise ValueError("event advance consumer must be callable")
    _bound_clock(binding)
    arm = (account, scenario, role)
    binding.event_context(*arm, 1)  # Unknown arms cannot qualify even an empty prefix.
    actor, reserve = _rows(actor_rows), _rows(closing_rows)
    actor_count = reserve_count = batches = event_count = rejected_count = 0
    tail_missing = False
    for advance in _advances(binding, arm, actor, reserve):
        actor_count += advance.phase == "actor"
        reserve_count += advance.phase == "reserve"
        rejected_count += advance.native_rejected
        if advance.events is None:
            tail_missing = True
        else:
            batches += 1
            event_count += len(advance.events)
        if on_advance is not None:
            on_advance(advance)
    return BoundEventRowCoverage(
        binding.digest,
        account,
        scenario,
        role,
        _ACTOR_ROWS,
        _RESERVE_ROWS,
        actor_count,
        reserve_count,
        batches,
        event_count,
        rejected_count,
        tail_missing,
    )
