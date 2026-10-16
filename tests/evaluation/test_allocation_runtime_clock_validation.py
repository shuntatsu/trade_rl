from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from tests.evaluation.test_allocation_rl_env import capability, parameters
from trade_rl.evaluation.objectives import BoundObjectiveClock
from trade_rl.evaluation.rl_allocation.binding import validate_runtime_binding
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.forecasts.training_trace import _timestamp


@pytest.fixture(scope="module")
def runtime_case():
    args = parameters()
    executor = MarketExecutor(
        args["dataset"], args["execution_cost"], insolvency_valuation="retain_debt"
    )
    return {
        "args": args,
        "executor": executor,
        "risk": args["risk_config"],
        "recipe": args["bound"].objective.deployment_recipe_digest,
    }


def _utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _bound_for_span(
    runtime_case,
    start: datetime,
    stop: datetime,
    step_seconds: int,
) -> BoundObjectiveClock:
    elapsed = stop - start
    assert elapsed.microseconds == 0
    horizon = elapsed.days * 86_400 + elapsed.seconds
    clock = replace(
        runtime_case["args"]["bound"].clock,
        decision_interval_seconds=step_seconds,
        execution_interval_seconds=step_seconds,
        reward_interval_seconds=step_seconds,
        economic_horizon_seconds=horizon,
        gamma=1.0,
        reward_schema="equity_delta_v1",
    )
    objective = replace(
        runtime_case["args"]["bound"].objective,
        evaluation_start=start,
        evaluation_stop_exclusive=stop,
    )
    return BoundObjectiveClock(objective, clock)


def _forged_dataset(timestamps: np.ndarray, *, n_bars: int | None = None):
    # Low-level timestamp cases intentionally bypass MarketDataset construction.
    return SimpleNamespace(
        timestamps=timestamps,
        n_bars=len(timestamps) if n_bars is None else n_bars,
    )


def _invoke(
    runtime_case,
    timestamps: np.ndarray,
    *,
    bound: BoundObjectiveClock | None = None,
    start_index: int = 0,
    stop_index: int = 1,
    n_bars: int | None = None,
    dataset: object | None = None,
    recipe: str | None = None,
):
    return validate_runtime_binding(
        _forged_dataset(timestamps, n_bars=n_bars) if dataset is None else dataset,
        runtime_case["args"]["bound"] if bound is None else bound,
        runtime_case["executor"],
        runtime_case["risk"],
        runtime_case["recipe"] if recipe is None else recipe,
        start_index=start_index,
        stop_index=stop_index,
    )


def _outcome(call):
    try:
        return call()
    except Exception as error:
        return type(error), str(error)


def _scalar_timestamp_oracle(timestamps: np.ndarray, bound: BoundObjectiveClock):
    """Retain the old scalar clock path and Python-int interval arithmetic."""
    try:
        times = tuple(_timestamp(value, field="episode_time") for value in timestamps)
        for actual, declared in (
            (times[0], bound.objective.evaluation_start),
            (times[-1], bound.objective.evaluation_stop_exclusive),
        ):
            expected = _timestamp(
                np.datetime64(declared.astimezone(UTC).replace(tzinfo=None), "us"),
                field="declared UTC endpoint",
            )
            if actual != expected:
                raise ValueError(
                    "Dataset episode endpoints differ from declared UTC horizon"
                )
        ticks = tuple(int(value.astype(np.int64)) for value in times)
        expected_gap_ns = bound.clock.execution_interval_seconds * 1_000_000_000
        if any(b - a != expected_gap_ns for a, b in zip(ticks, ticks[1:])):
            raise ValueError("Dataset intervals differ from declared financial clock")
    except Exception as error:
        return type(error), str(error)
    return None


def _profiled(call):
    calls: list[str | None] = []
    timestamp_code = _timestamp.__code__
    previous_profile = sys.getprofile()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is timestamp_code:
            calls.append(frame.f_locals.get("field"))

    try:
        sys.setprofile(profile)
        result = _outcome(call)
    finally:
        sys.setprofile(previous_profile)
    return result, calls


def _regular_ns(start: str, count: int, *, step_hours: int = 1) -> np.ndarray:
    return np.datetime64(start, "ns") + np.arange(count) * np.timedelta64(
        step_hours, "h"
    )


@pytest.mark.parametrize(
    ("unit", "origin", "step_count", "expected_start", "expected_stop"),
    (
        ("Y", "2022", 1, "2022-01-01", "2023-01-01"),
        ("M", "2026-01", 1, "2026-01-01", "2026-02-01"),
        # datetime64[W] is anchored to 1970-01-01 (Thursday), not Monday.
        ("W", "2026-01-05", 1, "2026-01-01", "2026-01-08"),
        ("D", "2026-01-01", 1, "2026-01-01", "2026-01-02"),
        ("2D", "2026-01-01", 1, "2026-01-01", "2026-01-03"),
        ("h", "2026-01-01T06", 1, "2026-01-01T06", "2026-01-01T07"),
        ("3h", "2026-01-01T06", 1, "2026-01-01T06", "2026-01-01T09"),
        ("m", "2026-01-01T06:00", 1, "2026-01-01T06:00", "2026-01-01T06:01"),
        ("s", "2026-01-01T06:00:00", 1, "2026-01-01T06:00:00", "2026-01-01T06:00:01"),
        (
            "ms",
            "2026-01-01T06:00:00",
            1_000,
            "2026-01-01T06:00:00",
            "2026-01-01T06:00:01",
        ),
        (
            "5ms",
            "2026-01-01T06:00:00",
            200,
            "2026-01-01T06:00:00",
            "2026-01-01T06:00:01",
        ),
        (
            "us",
            "2026-01-01T06:00:00",
            1_000_000,
            "2026-01-01T06:00:00",
            "2026-01-01T06:00:01",
        ),
        (
            "ns",
            "2026-01-01T06:00:00",
            1_000_000_000,
            "2026-01-01T06:00:00",
            "2026-01-01T06:00:01",
        ),
        (
            "ps",
            "1970-01-01T00:00:00",
            1_000_000_000_000,
            "1970-01-01T00:00:00",
            "1970-01-01T00:00:01",
        ),
        (
            "fs",
            "1970-01-01T00:00:00",
            1_000_000_000_000_000,
            "1970-01-01T00:00:00",
            "1970-01-01T00:00:01",
        ),
        (
            "as",
            "1970-01-01T00:00:00",
            1_000_000_000_000_000_000,
            "1970-01-01T00:00:00",
            "1970-01-01T00:00:01",
        ),
    ),
)
def test_exact_datetime64_units_match_scalar_and_utc_oracles(
    runtime_case,
    unit: str,
    origin: str,
    step_count: int,
    expected_start: str,
    expected_stop: str,
) -> None:
    start = np.datetime64(origin, unit)
    values = np.array(
        [start, start + np.timedelta64(step_count, unit)],
        dtype=f"datetime64[{unit}]",
    )
    start_utc, stop_utc = map(_utc_datetime, (expected_start, expected_stop))
    step = np.timedelta64(step_count, unit)
    assert values[1] == values[0] + step
    elapsed = stop_utc - start_utc
    assert elapsed.microseconds == 0
    step_seconds = elapsed.days * 86_400 + elapsed.seconds
    bound = _bound_for_span(runtime_case, start_utc, stop_utc, step_seconds)

    assert _scalar_timestamp_oracle(values, bound) is None
    assert _invoke(runtime_case, values, bound=bound) is None


def test_declared_non_utc_endpoints_are_compared_as_utc(runtime_case) -> None:
    local = timezone(timedelta(hours=-5))
    start = datetime(2026, 1, 1, 1, tzinfo=local)
    stop = datetime(2026, 1, 1, 3, tzinfo=local)
    values = _regular_ns("2026-01-01T06:00:00", 3)
    bound = _bound_for_span(runtime_case, start, stop, step_seconds=3_600)

    assert _scalar_timestamp_oracle(values, bound) is None
    assert (
        _invoke(
            runtime_case,
            values,
            bound=bound,
            stop_index=2,
        )
        is None
    )


@pytest.mark.parametrize(
    ("first", "second", "message"),
    (
        (
            np.datetime64(1, "ps"),
            np.datetime64("NaT", "ps"),
            "episode_time must be exactly representable in nanoseconds",
        ),
        (
            np.datetime64("NaT", "ps"),
            np.datetime64(1, "ps"),
            "episode_time must be a non-NaT timestamp",
        ),
    ),
)
def test_first_invalid_tick_keeps_scalar_error_precedence(
    runtime_case, first: np.datetime64, second: np.datetime64, message: str
) -> None:
    values = np.array([first, second], dtype="datetime64[ps]")
    bound = runtime_case["args"]["bound"]
    expected = _scalar_timestamp_oracle(values, bound)

    assert expected == (ValueError, message)
    assert _outcome(lambda: _invoke(runtime_case, values)) == expected


def test_out_of_range_coarse_tick_keeps_scalar_error(runtime_case) -> None:
    values = np.array(
        [np.datetime64("3000-01-01T00", "s"), np.datetime64("3000-01-01T01", "s")],
        dtype="datetime64[s]",
    )
    expected = _scalar_timestamp_oracle(values, runtime_case["args"]["bound"])

    assert expected == (
        ValueError,
        "episode_time must be exactly representable in nanoseconds",
    )
    assert _outcome(lambda: _invoke(runtime_case, values)) == expected


@pytest.mark.parametrize(
    ("wrong_endpoint", "expected_endpoint_calls"),
    (("start", 1), ("stop", 2)),
)
def test_endpoint_checks_remain_start_then_stop_before_cadence(
    runtime_case, wrong_endpoint: str, expected_endpoint_calls: int
) -> None:
    values = _regular_ns("2026-01-01T06:00:00", 3)
    if wrong_endpoint == "start":
        values[0] += np.timedelta64(1, "h")
    else:
        values[-1] += np.timedelta64(1, "h")
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 8, tzinfo=UTC),
        step_seconds=3_600,
    )

    expected = _scalar_timestamp_oracle(values, bound)
    assert expected == (
        ValueError,
        "Dataset episode endpoints differ from declared UTC horizon",
    )
    actual, fields = _profiled(
        lambda: _invoke(runtime_case, values, bound=bound, stop_index=2)
    )
    assert actual == expected
    assert fields.count("declared UTC endpoint") == expected_endpoint_calls


def test_bad_interior_tick_precedes_endpoint_mismatch(runtime_case) -> None:
    values = np.array(
        [
            np.datetime64("2026-01-01T07:00:00", "ns"),
            np.datetime64("NaT", "ns"),
            np.datetime64("2026-01-01T08:00:00", "ns"),
        ]
    )
    bound = runtime_case["args"]["bound"]
    expected = _scalar_timestamp_oracle(values, bound)

    assert expected == (ValueError, "episode_time must be a non-NaT timestamp")
    assert (
        _outcome(lambda: _invoke(runtime_case, values, bound=bound, stop_index=2))
        == expected
    )


def test_irregular_session_clock_is_checked_after_matching_endpoints(
    runtime_case,
) -> None:
    values = np.array(
        [
            np.datetime64("2026-01-01T06:00:00", "ns"),
            np.datetime64("2026-01-01T08:00:00", "ns"),
            np.datetime64("2026-01-01T09:00:00", "ns"),
        ]
    )
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 9, tzinfo=UTC),
        step_seconds=3_600,
    )
    expected = _scalar_timestamp_oracle(values, bound)

    assert expected == (
        ValueError,
        "Dataset intervals differ from declared financial clock",
    )
    assert (
        _outcome(lambda: _invoke(runtime_case, values, bound=bound, stop_index=2))
        == expected
    )


def test_profile_mismatch_still_precedes_malformed_timestamps(runtime_case) -> None:
    values = np.array([np.datetime64("NaT", "ns")] * 2)
    original = runtime_case["args"]["bound"]
    bad_objective = replace(original.objective, economics_digest="0" * 64)
    bad_bound = replace(original, objective=bad_objective)

    assert _outcome(lambda: _invoke(runtime_case, values, bound=bad_bound)) == (
        ValueError,
        "declared profile or action recipe differs from actual runtime",
    )


def test_each_call_revalidates_reused_mutable_timestamp_view(runtime_case) -> None:
    values = _regular_ns("2026-01-01T06:00:00", 2)
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 7, tzinfo=UTC),
        step_seconds=3_600,
    )
    dataset = _forged_dataset(values)

    assert _invoke(runtime_case, values, bound=bound, dataset=dataset) is None
    values[1] = np.datetime64("NaT", "ns")
    assert _outcome(
        lambda: _invoke(runtime_case, values, bound=bound, dataset=dataset)
    ) == (ValueError, "episode_time must be a non-NaT timestamp")


def test_public_environment_step_revalidates_current_dataset_timestamps() -> None:
    env = capability().AllocationTradingEnv(**parameters())
    env.reset(seed=7)
    timestamps = env.dataset.timestamps.copy()
    timestamps[7] = np.datetime64("NaT", "ns")
    # Replace the immutable backing array to model a changed runtime carrier.
    object.__setattr__(env.dataset, "timestamps", timestamps)

    with pytest.raises(ValueError, match="episode_time must be a non-NaT timestamp"):
        env.step(3)


def test_subclassed_clock_array_falls_back_to_scalar_validation(runtime_case) -> None:
    class TimestampArray(np.ndarray):
        pass

    values = _regular_ns("2026-01-01T06:00:00", 2).view(TimestampArray)
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 7, tzinfo=UTC),
        step_seconds=3_600,
    )
    actual, fields = _profiled(lambda: _invoke(runtime_case, values, bound=bound))

    assert actual is None
    assert fields == [
        "episode_time",
        "episode_time",
        "declared UTC endpoint",
        "declared UTC endpoint",
    ]


def test_object_dtype_clock_array_falls_back_to_scalar_validation(runtime_case) -> None:
    values = np.array(
        [
            np.datetime64("2026-01-01T06:00:00", "ns"),
            np.datetime64("2026-01-01T07:00:00", "ns"),
        ],
        dtype=object,
    )
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 7, tzinfo=UTC),
        step_seconds=3_600,
    )
    actual, fields = _profiled(lambda: _invoke(runtime_case, values, bound=bound))

    assert actual is None
    assert fields == [
        "episode_time",
        "episode_time",
        "declared UTC endpoint",
        "declared UTC endpoint",
    ]


@pytest.mark.parametrize("shape", ["two_dimensional", "non_datetime", "generic_nat"])
def test_unsupported_clock_shapes_and_dtypes_keep_scalar_failures(
    runtime_case, shape: str
) -> None:
    if shape == "two_dimensional":
        values = _regular_ns("2026-01-01T06:00:00", 2)[:, None]
    elif shape == "non_datetime":
        values = np.array(["2026-01-01T06", "2026-01-01T07"], dtype=str)
    else:
        values = np.array([np.datetime64("NaT", "generic")], dtype="datetime64")
    expected = _scalar_timestamp_oracle(values, runtime_case["args"]["bound"])

    assert expected is not None
    actual, calls = _profiled(lambda: _invoke(runtime_case, values, n_bars=2))
    assert actual == expected
    assert calls


def test_strided_and_big_endian_datetime_arrays_preserve_values(runtime_case) -> None:
    backing = _regular_ns("2026-01-01T06:00:00", 4)
    original = backing.copy()
    strided = backing[::3]
    assert not strided.flags.c_contiguous
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 9, tzinfo=UTC),
        step_seconds=10_800,
    )
    assert _scalar_timestamp_oracle(strided, bound) is None
    assert _invoke(runtime_case, strided, bound=bound) is None
    np.testing.assert_array_equal(backing, original)

    big_endian = backing[::3].astype(np.dtype("datetime64[ns]").newbyteorder(">"))
    assert _scalar_timestamp_oracle(big_endian, bound) is None
    assert _invoke(runtime_case, big_endian, bound=bound) is None


def test_gap_larger_than_int64_uses_exact_python_integer_arithmetic(
    runtime_case,
) -> None:
    gap_ns = 10_000_000_000_000_000_000
    ticks = np.array(
        [-9_000_000_000_000_000_000, 1_000_000_000_000_000_000],
        dtype=np.int64,
    )
    values = ticks.view("datetime64[ns]")
    assert gap_ns > np.iinfo(np.int64).max
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    start, stop = (epoch + timedelta(microseconds=int(tick) // 1_000) for tick in ticks)
    step_seconds = gap_ns // 1_000_000_000
    assert (stop - start).days * 86_400 + (stop - start).seconds == step_seconds
    bound = _bound_for_span(runtime_case, start, stop, step_seconds)

    assert _scalar_timestamp_oracle(values, bound) is None
    assert _invoke(runtime_case, values, bound=bound) is None


def test_long_plain_clock_uses_constant_scalar_timestamp_dispatch(runtime_case) -> None:
    count = 64
    values = _regular_ns("2026-01-01T06:00:00", count)
    bound = _bound_for_span(
        runtime_case,
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 6, tzinfo=UTC) + timedelta(hours=count - 1),
        step_seconds=3_600,
    )

    actual, fields = _profiled(
        lambda: _invoke(
            runtime_case,
            values,
            bound=bound,
            stop_index=count - 1,
            n_bars=count,
        )
    )
    assert actual is None
    assert len(fields) <= 2, (
        "plain runtime clock normalization should keep scalar dispatch constant "
        f"as episode length grows, saw {len(fields)} calls for {count} timestamps"
    )
