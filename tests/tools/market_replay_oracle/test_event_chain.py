"""Manufactured actor/reserve row coverage through the bound event reader."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import chain
from typing import Any

import pytest

import tools.market_replay_oracle as oracle
from tools.market_replay_oracle.identity import canonical

from .test_events import _event, _legacy
from .test_exact_fill_binding import _binding

ARM = ("H24_RIDGE_A1_BTCUSDT", "BASEL0", "nonrl")
ACTOR_ROWS = 6527
RESERVE_ROWS = 72
_MISSING = object()


def _api() -> Any:
    assert hasattr(oracle, "walk_bound_event_rows"), "bound event-row walker is missing"
    return getattr(oracle, "walk_bound_event_rows")


@pytest.fixture(scope="module")
def binding() -> Any:
    return _binding()


def _context(binding: Any, count: int) -> Any:
    account, scenario, role = ARM
    return binding.event_context(account, scenario, role, count)


def _clock(binding: Any, count: int) -> dict[str, int]:
    # Independent test oracle: use the manufactured raw clock and literal frozen
    # start, not binding.progress() or event_context() used by the implementation.
    processing_index = 2903 + count
    timestamps = binding.content.arrays["timestamps"]
    return {
        "decision_index": processing_index - 1,
        "decision_time_ns": int(timestamps[processing_index - 1].astype("int64")),
        "processing_index": processing_index,
        "processing_time_ns": int(timestamps[processing_index].astype("int64")),
        "symbol_index": 0,
    }


def _actor_row(
    binding: Any,
    count: int,
    events: object = (),
    *,
    rejected: object = None,
) -> dict[str, Any]:
    clock = _clock(binding, count)
    native_events = (
        _MISSING
        if events is _MISSING
        else (list(events) if isinstance(events, (list, tuple)) else events)
    )
    return {
        "decision_index": clock["decision_index"],
        "decision_time_ns": clock["decision_time_ns"],
        "native_validation_failure": rejected,
        "native": {
            "decision_index": clock["decision_index"],
            "decision_time_ns": clock["decision_time_ns"],
            "processing_index": clock["processing_index"],
            "processing_time_ns": clock["processing_time_ns"],
            "symbol_index": clock["symbol_index"],
            "execution": {
                "next_index": clock["processing_index"],
                "bars_advanced": 1,
            },
            "order_events": native_events,
        },
    }


def _reserve_row(
    binding: Any,
    count: int,
    events: object = (),
    *,
    clock_valid: object = True,
) -> dict[str, Any]:
    clock = _clock(binding, count)
    native_events = (
        _MISSING
        if events is _MISSING
        else (list(events) if isinstance(events, (list, tuple)) else events)
    )
    return {
        "start_index": clock["decision_index"],
        "processing_index": clock["processing_index"],
        "processing_time_ns": clock["processing_time_ns"],
        "clock_valid": clock_valid,
        "native": {
            "execution": {
                "next_index": clock["processing_index"],
                "bars_advanced": 1,
            },
            "order_events": native_events,
        },
    }


def _actor_stream(binding: Any, *, rejected_at: int | None = None):
    for count in range(1, ACTOR_ROWS + 1):
        rejected = (
            {"error_type": "ValueError", "message": "retained rejected fact"}
            if count == rejected_at
            else None
        )
        yield _actor_row(binding, count, rejected=rejected)


def _reserve_stream(binding: Any, *, extra: bool = False):
    stop = ACTOR_ROWS + RESERVE_ROWS + int(extra)
    for count in range(ACTOR_ROWS + 1, stop + 1):
        yield _reserve_row(binding, count)


def _walk(
    binding: Any,
    *,
    actor_rows: Any = (),
    closing_rows: Any = (),
    on_advance: Any = None,
) -> Any:
    account, scenario, role = ARM
    return _api()(
        binding,
        account=account,
        scenario=scenario,
        role=role,
        actor_rows=actor_rows,
        closing_rows=closing_rows,
        on_advance=on_advance,
    )


def _for_context(event: dict[str, Any], context: Any) -> dict[str, Any]:
    event = dict(event)
    event.update(
        dataset_id=context.dataset_id,
        execution_policy_digest=context.execution_policy_digest,
        symbol_index=context.symbol_index,
        processing_index=context.processing_index,
        timestamp_ns=context.processing_time_ns,
    )
    return event


def test_row_walker_public_api_is_available() -> None:
    assert callable(_api())


def test_full_schedule_streams_actor_then_reserve_and_reports_row_coverage(
    binding: Any,
) -> None:
    seen = 0
    boundaries: dict[int, tuple[str, int, bool]] = {}

    def observe(advance: Any) -> None:
        nonlocal seen
        seen += 1
        if advance.count in (1, ACTOR_ROWS, ACTOR_ROWS + 1, ACTOR_ROWS + RESERVE_ROWS):
            boundaries[advance.count] = (
                advance.phase,
                advance.context.processing_index,
                advance.native_rejected,
            )
        assert advance.events == ()

    result = _walk(
        binding,
        actor_rows=_actor_stream(binding, rejected_at=1),
        closing_rows=_reserve_stream(binding),
        on_advance=observe,
    )

    assert result.binding_digest == binding.digest
    assert (result.account, result.scenario, result.role) == ARM
    assert (result.expected_actor_rows, result.expected_reserve_rows) == (
        ACTOR_ROWS,
        RESERVE_ROWS,
    )
    assert (result.observed_actor_rows, result.observed_reserve_rows) == (
        ACTOR_ROWS,
        RESERVE_ROWS,
    )
    assert result.parsed_event_batches == ACTOR_ROWS + RESERVE_ROWS
    assert result.event_count == 0
    assert result.rejected_native_rows == 1
    assert result.tail_native_missing is False
    assert result.all_expected_batches_present is True
    assert result.scope == "BOUND_EVENT_ROW_COVERAGE_ONLY"
    assert seen == ACTOR_ROWS + RESERVE_ROWS
    assert boundaries == {
        1: ("actor", 2904, True),
        ACTOR_ROWS: ("actor", 9430, False),
        ACTOR_ROWS + 1: ("reserve", 9431, False),
        ACTOR_ROWS + RESERVE_ROWS: ("reserve", 9502, False),
    }


def test_present_empty_event_array_counts_as_a_parsed_zero_event_batch(
    binding: Any,
) -> None:
    result = _walk(binding, actor_rows=(_actor_row(binding, 1, events=[]),))

    assert result.observed_actor_rows == 1
    assert result.parsed_event_batches == 1
    assert result.event_count == 0
    assert result.all_expected_batches_present is False
    assert result.tail_native_missing is False


def test_callback_consumes_the_actor_stream_one_row_at_a_time(binding: Any) -> None:
    produced = 0
    consumed = 0

    def rows():
        nonlocal produced
        for count in range(1, 4):
            assert produced == consumed
            produced += 1
            yield _actor_row(binding, count)

    def observe(_: Any) -> None:
        nonlocal consumed
        consumed += 1

    result = _walk(binding, actor_rows=rows(), on_advance=observe)

    assert (produced, consumed) == (3, 3)
    assert result.observed_actor_rows == 3
    assert result.parsed_event_batches == 3
    assert result.all_expected_batches_present is False


def test_missing_event_array_is_not_normalized_to_an_empty_batch(binding: Any) -> None:
    row = _actor_row(binding, 1)
    row["native"].pop("order_events")

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(row,))


def test_explicit_trailing_actor_native_null_is_prefix_only(binding: Any) -> None:
    row = _actor_row(binding, 1)
    row["native"] = None
    observed: list[Any] = []

    result = _walk(binding, actor_rows=(row,), on_advance=observed.append)

    assert result.observed_actor_rows == 1
    assert result.parsed_event_batches == 0
    assert result.tail_native_missing is True
    assert result.all_expected_batches_present is False
    assert len(observed) == 1
    assert observed[0].events is None
    assert observed[0].native_rejected is False


def test_native_null_cannot_be_followed_by_actor_or_reserve_rows(binding: Any) -> None:
    first = _actor_row(binding, 1)
    first["native"] = None
    second = _actor_row(binding, 2)
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(first, second))

    reserve = _reserve_row(binding, ACTOR_ROWS + 1)
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(first,), closing_rows=(reserve,))


@pytest.mark.parametrize("ordering", ["duplicate", "gap", "reversed"])
def test_actor_rows_must_be_a_contiguous_ordered_prefix(
    binding: Any, ordering: str
) -> None:
    first, second, third = (_actor_row(binding, count) for count in (1, 2, 3))
    rows = {
        "duplicate": (first, first),
        "gap": (first, third),
        "reversed": (second, first),
    }[ordering]

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=rows)


@pytest.mark.parametrize(
    "path,value",
    [
        (("decision_index",), True),
        (("decision_time_ns",), -1),
        (("native", "decision_index"), 1),
        (("native", "decision_time_ns"), -1),
        (("native", "processing_index"), 9430),
        (("native", "processing_time_ns"), -1),
        (("native", "symbol_index"), 1),
        (("native", "execution", "next_index"), 9999),
        (("native", "execution", "bars_advanced"), 2),
    ],
)
def test_actor_outer_and_native_clocks_match_bound_context(
    binding: Any, path: tuple[str, ...], value: object
) -> None:
    row = _actor_row(binding, 1)
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(row,))


def test_coherently_shifted_actor_clocks_still_differ_from_the_bound_clock(
    binding: Any,
) -> None:
    row = _actor_row(binding, 1)
    one_hour_ns = 3_600_000_000_000
    row["decision_time_ns"] += one_hour_ns
    row["native"]["decision_time_ns"] += one_hour_ns
    row["native"]["processing_time_ns"] += one_hour_ns

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(row,))


@pytest.mark.parametrize(
    "path,value",
    [
        (("start_index",), 1),
        (("processing_index",), 9430),
        (("processing_time_ns",), -1),
        (("native", "execution", "next_index"), 9999),
        (("native", "execution", "bars_advanced"), 2),
    ],
)
def test_reserve_outer_and_native_clocks_match_bound_context(
    binding: Any, path: tuple[str, ...], value: object
) -> None:
    row = _reserve_row(binding, ACTOR_ROWS + 1)
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=_actor_stream(binding), closing_rows=(row,))


def test_partial_actor_prefix_cannot_claim_or_start_reserve_coverage(
    binding: Any,
) -> None:
    reserve = _reserve_row(binding, ACTOR_ROWS + 1)

    with pytest.raises(ValueError):
        _walk(
            binding,
            actor_rows=(_actor_row(binding, 1),),
            closing_rows=(reserve,),
        )


def test_actor_and_reserve_phase_limits_reject_extra_rows(binding: Any) -> None:
    with pytest.raises(ValueError):
        _walk(
            binding,
            actor_rows=chain(
                _actor_stream(binding), (_actor_row(binding, ACTOR_ROWS + 1),)
            ),
        )

    with pytest.raises(ValueError):
        _walk(
            binding,
            actor_rows=_actor_stream(binding),
            closing_rows=_reserve_stream(binding, extra=True),
        )


@pytest.mark.parametrize(
    "account,scenario,role",
    [
        ("H24_RIDGE_A1_OTHER", "BASEL0", "nonrl"),
        (ARM[0], "FOREIGN", "nonrl"),
        (ARM[0], ARM[1], "invented"),
    ],
)
def test_unknown_arm_is_rejected_even_for_empty_prefix(
    binding: Any, account: str, scenario: str, role: str
) -> None:
    with pytest.raises(ValueError):
        _api()(
            binding,
            account=account,
            scenario=scenario,
            role=role,
            actor_rows=(),
            closing_rows=(),
        )


def test_bound_v2_fill_uses_static_symbol_clock_policy_and_lot(binding: Any) -> None:
    context = _context(binding, 1)
    fill = _for_context(
        _event(
            filled_lot_count=3,
            lot_size=0.125,
            filled_quantity=0.375,
            remaining_quantity=0.625,
            filled_notional=37.5,
        ),
        context,
    )
    observed: list[Any] = []

    result = _walk(
        binding,
        actor_rows=(_actor_row(binding, 1, events=[fill]),),
        on_advance=observed.append,
    )

    assert result.event_count == 1
    assert observed[0].events[0].exact_filled_quantity == Fraction(3, 8)


def test_first_reserve_zero_lot_accepts_v1_fill_with_its_bound_context(
    binding: Any,
) -> None:
    count = ACTOR_ROWS + 1
    context = _context(binding, count)
    fill = _for_context(_legacy(), context)
    observed: list[Any] = []

    result = _walk(
        binding,
        actor_rows=_actor_stream(binding),
        closing_rows=(_reserve_row(binding, count, events=[fill]),),
        on_advance=observed.append,
    )

    assert result.observed_actor_rows == ACTOR_ROWS
    assert result.observed_reserve_rows == 1
    assert result.parsed_event_batches == ACTOR_ROWS + 1
    assert observed[-1].count == count
    assert observed[-1].phase == "reserve"
    assert observed[-1].events[0].exact_filled_quantity == Fraction(3, 10)
    assert result.all_expected_batches_present is False


@pytest.mark.parametrize(
    "change",
    [
        {"symbol_index": 1},
        {"execution_policy_digest": "f" * 64},
        {"timestamp_ns": 1},
        {"sequence": 1},
    ],
)
def test_event_parser_rejects_batch_fields_outside_bound_context(
    binding: Any, change: dict[str, Any]
) -> None:
    context = _context(binding, 1)
    fill = _for_context(
        _event(
            filled_lot_count=3,
            lot_size=0.125,
            filled_quantity=0.375,
            remaining_quantity=0.625,
            filled_notional=37.5,
        ),
        context,
    )
    fill.update(change)

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(_actor_row(binding, 1, events=[fill]),))


@pytest.mark.parametrize(
    "row_change",
    [
        {"native_validation_failure": {"error_type": "ValueError"}},
        {"native_validation_failure": "failed"},
    ],
)
def test_actor_rejection_marker_has_closed_shape(
    binding: Any, row_change: dict[str, Any]
) -> None:
    row = _actor_row(binding, 1)
    row.update(row_change)

    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(row,))


def test_reserve_clock_validity_is_reported_as_rejected_not_authorized(
    binding: Any,
) -> None:
    observed: list[Any] = []
    result = _walk(
        binding,
        actor_rows=_actor_stream(binding),
        closing_rows=(_reserve_row(binding, ACTOR_ROWS + 1, clock_valid=False),),
        on_advance=observed.append,
    )

    assert result.observed_reserve_rows == 1
    assert result.rejected_native_rows == 1
    assert result.all_expected_batches_present is False
    assert observed[-1].native_rejected is True


def test_missing_native_fields_and_non_array_batches_reject(binding: Any) -> None:
    actor = _actor_row(binding, 1)
    actor.pop("native")
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(actor,))

    actor = _actor_row(binding, 1, events=None)
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(actor,))

    reserve = _reserve_row(binding, ACTOR_ROWS + 1)
    reserve.pop("native")
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=_actor_stream(binding), closing_rows=(reserve,))


def test_callback_errors_propagate_and_noncallable_callbacks_reject(
    binding: Any,
) -> None:
    def fail(_: Any) -> None:
        raise RuntimeError("consumer stopped")

    with pytest.raises(RuntimeError, match="consumer stopped"):
        _walk(binding, actor_rows=(_actor_row(binding, 1),), on_advance=fail)

    with pytest.raises((TypeError, ValueError)):
        _walk(binding, actor_rows=(_actor_row(binding, 1),), on_advance=object())


@pytest.mark.parametrize("rows", [None, {}, "", b""])
def test_malformed_row_containers_are_not_empty_prefixes(
    binding: Any, rows: Any
) -> None:
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=rows)


def test_row_coverage_and_callback_facts_are_immutable(binding: Any) -> None:
    observed: list[Any] = []
    result = _walk(
        binding, actor_rows=(_actor_row(binding, 1),), on_advance=observed.append
    )
    with pytest.raises(FrozenInstanceError):
        result.event_count = 99
    with pytest.raises(FrozenInstanceError):
        observed[0].count = 99


def test_missing_rejection_marker_and_rejected_null_native_are_invalid(
    binding: Any,
) -> None:
    row = _actor_row(binding, 1)
    row.pop("native_validation_failure")
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(row,))
    row = _actor_row(
        binding, 1, rejected={"error_type": "ValueError", "message": "bad"}
    )
    row["native"] = None
    with pytest.raises(ValueError):
        _walk(binding, actor_rows=(row,))


def test_non_boolean_reserve_clock_marker_is_invalid(binding: Any) -> None:
    with pytest.raises(ValueError):
        _walk(
            binding,
            actor_rows=_actor_stream(binding),
            closing_rows=(_reserve_row(binding, ACTOR_ROWS + 1, clock_valid=1),),
        )


def test_row_walker_rejects_retuned_clock_in_public_binding(binding: Any) -> None:
    payload = binding.payload()
    payload["clock"]["actor_start"] = 2904
    changed = replace(binding, _canonical_payload=canonical(payload))
    with pytest.raises(ValueError):
        _walk(changed)


def test_every_frozen_account_scenario_role_accepts_its_own_row_context(
    binding: Any,
) -> None:
    arms = binding.payload()["schedule"]["arms"]
    assert len(arms) == 48
    for arm in arms:
        row = _actor_row(binding, 1)
        row["native"]["symbol_index"] = arm["symbol_index"]
        result = _api()(
            binding,
            account=arm["account"],
            scenario=arm["scenario"],
            role=arm["role"],
            actor_rows=(row,),
            closing_rows=(),
        )
        assert (result.account, result.scenario, result.role) == (
            arm["account"],
            arm["scenario"],
            arm["role"],
        )
        assert result.parsed_event_batches == 1
        assert result.all_expected_batches_present is False
