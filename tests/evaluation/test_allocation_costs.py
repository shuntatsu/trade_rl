"""Literal synthetic cost and refusal oracles; no market calibration evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import numpy as np
import pytest

from tests.strategies.test_simple_return_stream import time
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    VolumeUnit,
)
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
)


def api():
    name = "trade_rl.evaluation.allocation_costs"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail("Missing declared allocation horizon-cost producer", pytrace=False)


def market(*, future=False):
    times = time(0) + np.arange(72) * np.timedelta64(1, "h")
    series = {}
    for symbol in ("ALPHA", "BETA"):
        close = np.full(72, 100.0 if symbol == "ALPHA" else 50.0)
        volume = np.full(72, 100_000.0)
        rates, counts = np.zeros(72), np.zeros(72, dtype=np.int32)
        rates[8], rates[16] = 0.0001, 0.0005
        counts[8], counts[16] = 1, 2
        if future:
            close[25:] *= 7
            volume[25:] *= 11
            rates[25:] = -0.05
            counts[25:] = 6
        series[symbol] = RawMarketSeries(
            timestamps=times,
            open=close,
            high=1.25 * close,
            low=0.75 * close,
            close=close,
            volume=volume,
            funding_rate=rates,
            tradable=np.ones(72, dtype=np.bool_),
            funding_available=counts > 0,
            funding_event_count=counts,
        )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="range", kind=FeatureKind.HIGH_LOW_RANGE),),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource(series),
        tuple(
            InstrumentContract(
                s,
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                volume_unit=VolumeUnit.QUOTE_NOTIONAL,
            )
            for s in series
        ),
        execution_economics=ExecutionEconomicsProfile(
            name="synthetic explicit assumptions",
            taker_fee_rate=0.002,
            spread_rate=0.004,
            max_participation_rate=0.03,
            borrow_rate=0.1095,
        ),
    )


def execution_cost(**changes):
    return replace(
        ExecutionCostConfig(
            fee_rate=0.001,
            taker_fee_rate=0.0007,
            spread_rate=0.003,
            impact_rate=0.05,
            multiplier=2,
            max_participation_rate=0.04,
            slippage_std=0.004 * math.sqrt(math.pi / 2),
            tail_slippage_probability=0.25,
            tail_slippage_multiplier=3,
            borrow_rate_multiplier=1.5,
            processing_bar_volume_capacity=False,
            trigger_volume_fractions=(0.5, 0.5, 0.25, 0.0),
        ),
        **changes,
    )


def recipe(**changes):
    cost = changes.pop("execution_cost", execution_cost())
    arguments = dict(
        execution_cost=cost,
        expected_execution_policy_digest=cost.execution_policy_digest,
        reference_notional=1000.0,
        assumptions_identity="synthetic economics declaration",
        assumptions_available_at=time(0),
        fee_assumption="Configured and Dataset generic/taker rates add; maker zero.",
        spread_assumption="Synthetic same-condition full taker spread.",
        impact_assumption="Synthetic reference-size square-root participation proxy.",
        slippage_assumption="Absolute-normal tail mixture expectation, no RNG draw.",
        borrow_assumption="Synthetic annual 365-day borrow convention.",
        funding_source_identity="synthetic complete aggregated funding source",
        funding_coverage_start=time(0),
        funding_coverage_stop=time(72),
        funding_lookback_hours=24,
        funding_minimum_event_count=3,
        funding_maximum_counted_bar_close_gap_hours=24.0,
        funding_maximum_counted_bar_close_age_hours=24.0,
        cash_zero_rationale="Research quote cash accrues no interest.",
    )
    return api().DeclaredAllocationCostRecipe(**(arguments | changes))


def estimate(dataset=None, supplied=None, *, decisions=(24,), horizon=48):
    return api().estimate_declared_horizon_costs(
        market() if dataset is None else dataset,
        decision_indices=decisions,
        horizon_hours=horizon,
        recipe=recipe() if supplied is None else supplied,
    )


def rates(cost):
    return tuple(
        getattr(cost, field)
        for field in (
            "buy_cost",
            "sell_cost",
            "exit_cost",
            "funding_return",
            "borrow_return",
            "cash_return",
        )
    )


def test_explicit_profile_requires_expected_dataset_id_before_arithmetic(monkeypatch):
    dataset = market()
    profile = ExecutionEconomicsProfile.from_payload(
        parse_identity_json(dataset.identity_payload_json)["execution_economics"]
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unbound profile reached cost arithmetic")

    monkeypatch.setattr(type(dataset), "market_notional", forbidden)
    with pytest.raises(ValueError, match="together"):
        api().estimate_declared_horizon_costs(
            dataset,
            decision_indices=(24,),
            horizon_hours=48,
            recipe=recipe(),
            declared_execution_economics=profile,
        )


def wrapped_market():
    dataset = market()
    identity = parse_identity_json(dataset.identity_payload_json)
    profile = ExecutionEconomicsProfile.from_payload(identity["execution_economics"])
    wrapped = replace(dataset, identity_payload_json=None).with_content_identity(
        {
            "schema": "synthetic_consumed_development_wrapper",
            "source_dataset": {
                "dataset_id": dataset.dataset_id,
                "identity_payload": identity,
            },
        }
    )
    return wrapped, profile


def explicit_estimate(dataset, profile, supplied=None, **changes):
    return api().estimate_declared_horizon_costs(
        dataset,
        **(
            dict(
                decision_indices=(24,),
                horizon_hours=48,
                recipe=recipe() if supplied is None else supplied,
                declared_execution_economics=profile,
                expected_dataset_id=dataset.dataset_id,
            )
            | changes
        ),
    )


def test_legacy_source_digest_and_explicit_mode_distinguish_equal_rates():
    dataset, supplied = market(), recipe()
    profile = ExecutionEconomicsProfile.from_payload(
        parse_identity_json(dataset.identity_payload_json)["execution_economics"]
    )
    legacy = estimate(dataset, supplied)
    explicit = explicit_estimate(dataset, profile, supplied)
    # Independent pre-change field roster: no producer declarations/hash helper.
    declarations = {
        name: getattr(supplied, name)
        for name in (
            "expected_execution_policy_digest",
            "reference_notional",
            "assumptions_identity",
            "fee_assumption",
            "spread_assumption",
            "impact_assumption",
            "slippage_assumption",
            "borrow_assumption",
            "funding_source_identity",
            "funding_lookback_hours",
            "funding_minimum_event_count",
            "funding_maximum_counted_bar_close_gap_hours",
            "funding_maximum_counted_bar_close_age_hours",
            "cash_zero_rationale",
        )
    }
    declarations["execution_cost"] = supplied.execution_cost.execution_policy_payload()
    for name in (
        "assumptions_available_at",
        "funding_coverage_start",
        "funding_coverage_stop",
    ):
        declarations[name] = int(getattr(supplied, name).astype(np.int64))
    body = {
        "dataset_id": dataset.dataset_id,
        "execution_economics": profile.to_payload(),
        "recipe": declarations,
        "symbol": "ALPHA",
        "decision_time": int(time(24).astype(np.int64)),
        "horizon_hours": 48,
    }
    for old, new in zip(legacy, explicit):
        body["symbol"] = old.symbol
        raw = json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert old.source_identity == hashlib.sha256(raw.encode()).hexdigest()
        explicit_body = body | {"profile_origin": "caller_declared_simulation_v1"}
        raw = json.dumps(
            explicit_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert new.source_identity == hashlib.sha256(raw.encode()).hexdigest()
    omitted = explicit_estimate(dataset, None, supplied, expected_dataset_id=None)
    assert [c.payload() for c in omitted] == [c.payload() for c in legacy]
    assert [rates(c) for c in legacy] == [rates(c) for c in explicit]
    assert all(a.source_identity != b.source_identity for a, b in zip(legacy, explicit))


def test_wrapped_profile_keeps_original_identity_arrays_and_literal_costs():
    dataset, profile = wrapped_market()
    dataset_id, identity_json = dataset.dataset_id, dataset.identity_payload_json
    arrays = {name: values.copy() for name, values in dataset.identity_arrays().items()}
    with pytest.raises(ValueError, match="JSON object"):
        estimate(dataset)
    supplied = recipe()
    original_recipe = repr(supplied)
    costs = explicit_estimate(dataset, profile, supplied)
    for cost in costs:
        assert rates(cost) == pytest.approx((0.0434, 0.0434, 0.0434, 0.0012, 0.0009, 0))
    assert (
        dataset.dataset_id == dataset_id
        and dataset.identity_payload_json == identity_json
    )
    assert dataset.identity_verified and repr(supplied) == original_recipe
    for name, expected in arrays.items():
        np.testing.assert_array_equal(dataset.identity_arrays()[name], expected)
    renamed = explicit_estimate(dataset, replace(profile, name="different declaration"))
    assert rates(renamed[0]) == rates(costs[0])
    assert renamed[0].source_identity != costs[0].source_identity


@pytest.mark.parametrize("row", (0, 71))
@pytest.mark.parametrize(
    "field,value",
    (
        ("fee_rate", 0.001),
        ("maker_fee_rate", 0.001),
        ("taker_fee_rate", 0.001),
        ("spread_rate", 0.001),
        ("max_participation_rate", 0.01),
        ("borrow_available", False),
        ("borrow_rate", 0.01),
    ),
)
def test_explicit_profile_rejects_whole_array_mismatch_before_arithmetic(
    monkeypatch, row, field, value
):
    dataset, profile = wrapped_market()
    values = dataset.resolved_array(field).copy()
    values[row, 1] = value
    identity = parse_identity_json(dataset.identity_payload_json)
    dataset = replace(
        dataset, **{field: values}, identity_payload_json=None
    ).with_content_identity(identity)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Whole-array disagreement reached cost arithmetic")

    monkeypatch.setattr(type(dataset), "market_notional", forbidden)
    with pytest.raises(ValueError, match="whole Dataset arrays"):
        explicit_estimate(dataset, profile)


@pytest.mark.parametrize(
    "failure",
    (
        "id_only",
        "bad_type",
        "bad_id",
        "bool_id",
        "wrong_id",
        "unverified",
        "stale_content",
        "conflict",
        "malformed",
        "malformed_empty",
        "malformed_bool",
        "forged_profile",
    ),
)
def test_explicit_profile_invalid_binding_before_arithmetic(monkeypatch, failure):
    dataset, profile = wrapped_market()
    changes = {}
    if failure == "id_only":
        profile = None
    elif failure == "bad_type":
        profile = profile.to_payload()
    elif failure == "bad_id":
        changes["expected_dataset_id"] = "not a digest"
    elif failure == "bool_id":
        changes["expected_dataset_id"] = False
    elif failure == "wrong_id":
        changes["expected_dataset_id"] = "0" * 64
    elif failure == "unverified":
        dataset = replace(dataset, identity_payload_json=None)
    elif failure == "stale_content":
        close = dataset.close.copy()
        close[0, 0] += 1
        object.__setattr__(dataset, "close", close)
    elif failure in {"conflict", "malformed", "malformed_empty", "malformed_bool"}:
        identity = parse_identity_json(dataset.identity_payload_json)
        identity["execution_economics"] = (
            replace(profile, name="conflicting existing declaration").to_payload()
            if failure == "conflict"
            else {"malformed": None, "malformed_empty": {}, "malformed_bool": False}[
                failure
            ]
        )
        dataset = replace(dataset, identity_payload_json=None).with_content_identity(
            identity
        )
    elif failure == "forged_profile":
        object.__setattr__(profile, "spread_rate", float("nan"))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid profile binding reached cost arithmetic")

    monkeypatch.setattr(type(dataset), "market_notional", forbidden)
    with pytest.raises(ValueError):
        explicit_estimate(dataset, profile, **changes)


def test_literal_additive_execution_carry_and_total_turnover_participation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Declared expectation consumed random numbers")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    supplied = recipe()
    costs = estimate(supplied=supplied, decisions=(25, 24))
    assert [(c.symbol, c.decision_time) for c in costs] == [
        ("ALPHA", time(24)),
        ("ALPHA", time(25)),
        ("BETA", time(24)),
        ("BETA", time(25)),
    ]
    assert rates(costs[0]) == pytest.approx((0.0434, 0.0434, 0.0434, 0.0012, 0.0009, 0))
    assert costs[0].horizon_end == time(72)
    assert len(costs[1].source_identity) == 64
    assert all(
        field in costs[1].payload()
        for field in (
            "buy_cost",
            "sell_cost",
            "exit_cost",
            "funding_return",
            "borrow_return",
            "cash_return",
        )
    )
    # The 0.03 pool cap and OPEN .5 give capacity 1500; participation remains .01.
    assert rates(estimate(supplied=recipe(reference_notional=1500))[0])[
        0
    ] == pytest.approx(2 * (0.0037 + 0.007 + 0.05 * math.sqrt(0.015) + 0.006))
    with pytest.raises(ValueError, match="capacity"):
        estimate(supplied=recipe(reference_notional=1501))


@pytest.mark.parametrize(
    "current,target,expected",
    (
        (0, 0.5, -0.044),
        (0.5, 0.5, -0.0223),
        (0.5, 0, -0.0217),
        (0, -0.5, -0.04325),
        (-0.5, -0.5, -0.02155),
        (-0.5, 0, -0.0217),
        (0.5, -0.5, -0.06495),
        (-0.5, 0.5, -0.0657),
        (0, 0, 0),
    ),
)
def test_literal_signed_entry_hold_flat_reversals_and_cash(current, target, expected):
    cost = estimate()[0]
    assert objective(cost, current, target) == pytest.approx(expected)


def objective(cost, current, target):
    arguments = cost._arguments() if hasattr(cost, "_arguments") else cost
    inputs = AllocationInputs(**arguments, expected_simple_return=0)
    context = AllocationContext(
        "synthetic account",
        "ALPHA",
        time(24),
        "a" * 64,
        current,
        "0",
        1000,
        1000,
        "0",
    )
    return (
        AfterCostTargetAllocator(lower_weight=target, upper_weight=target)
        .propose(inputs, context)
        .objective_value
    )


@pytest.mark.parametrize(
    "current,target,expected",
    (
        (0, 0.5, -0.01),
        (0.5, 0.5, -0.005),
        (0.5, 0, 0.05),
        (0, -0.5, 0.06),
        (-0.5, -0.5, 0.07),
        (-0.5, 0, 0.055),
        (0.5, -0.5, 0.05),
        (-0.5, 0.5, -0.015),
        (0, 0, 0.06),
    ),
)
def test_six_distinct_abstract_coefficients_detect_field_swaps(
    current, target, expected
):
    # Abstract allocator arithmetic only; nonzero cash is inadmissible to producer.
    cost = dict(
        symbol="ALPHA",
        decision_time=time(24),
        available_at=time(24),
        horizon_end=time(72),
        source_identity="abstract arithmetic oracle",
        buy_cost=0.01,
        sell_cost=0.02,
        exit_cost=0.03,
        funding_return=0.04,
        borrow_return=0.05,
        cash_return=0.06,
    )
    assert objective(cost, current, target) == pytest.approx(expected)


def test_future_suffix_changes_identity_without_changing_earlier_numeric_rates():
    a, b = estimate(market()), estimate(market(future=True))
    assert rates(a[0]) == rates(b[0])
    assert a[0].source_identity != b[0].source_identity
    negative = market()
    identity = parse_identity_json(negative.identity_payload_json)
    funding = -negative.funding_rate
    negative = replace(negative, funding_rate=funding, identity_payload_json=None)
    negative = negative.with_content_identity(identity)
    assert rates(estimate(negative)[0])[3] == pytest.approx(-0.0012)


def test_complete_window_excludes_left_boundary_and_execution_stress_excludes_carry():
    supplied = recipe(funding_lookback_hours=16, funding_minimum_event_count=2)
    assert estimate(supplied=supplied)[0].funding_return == pytest.approx(0.0015)
    stressed = estimate(supplied=recipe(execution_cost=execution_cost(multiplier=4)))[0]
    assert rates(stressed) == pytest.approx((0.0868, 0.0868, 0.0868, 0.0012, 0.0009, 0))


@pytest.mark.parametrize(
    "changes",
    (
        {"expected_execution_policy_digest": "0" * 64},
        {"reference_notional": 0},
        {"reference_notional": True},
        {"funding_lookback_hours": True},
        {"funding_minimum_event_count": 0},
        {"cash_zero_rationale": ""},
        {"fee_assumption": ""},
        {"funding_source_identity": ""},
        {"assumptions_available_at": np.datetime64("NaT")},
        {"funding_coverage_stop": time(0)},
        {"execution_cost": execution_cost(order_type="limit")},
        {"execution_cost": execution_cost(order_latency_bars=1)},
        {"execution_cost": execution_cost(processing_bar_volume_capacity=True)},
    ),
)
def test_invalid_recipe_rejected(changes):
    with pytest.raises(ValueError):
        recipe(**changes)


def test_mutable_trigger_alias_rejected_before_policy_digest(monkeypatch):
    cost = execution_cost(trigger_volume_fractions=[0.5, 0.5, 0.25, 0.0])
    cls = api().DeclaredAllocationCostRecipe
    arguments = {name: getattr(recipe(), name) for name in cls.__dataclass_fields__}
    arguments["execution_cost"] = cost

    def forbidden(_self):
        pytest.fail("Mutable configuration reached policy digest validation")

    monkeypatch.setattr(
        ExecutionCostConfig, "execution_policy_digest", property(forbidden)
    )
    with pytest.raises(ValueError, match="tuple"):
        cls(**arguments)


def test_recipe_and_returned_config_resist_ordinary_mutation():
    supplied = recipe()
    with pytest.raises(FrozenInstanceError):
        supplied.reference_notional = 1
    with pytest.raises(FrozenInstanceError):
        supplied.execution_cost.multiplier = 99
    with pytest.raises(TypeError):
        supplied.execution_cost.trigger_volume_fractions[0] = 0


@pytest.mark.parametrize(
    "failure",
    (
        "profile_missing",
        "profile_unknown",
        "profile_disagreement",
        "cash",
        "units",
        "mark",
        "history",
        "coverage",
        "future_source",
        "funding_count",
        "funding_due",
        "funding_rate_without_events",
        "unavailable_interval",
        "gap",
        "age",
    ),
)
def test_bad_source_rejected(failure):
    dataset, supplied = market(), recipe()
    identity = parse_identity_json(dataset.identity_payload_json)
    if failure == "profile_missing":
        identity.pop("execution_economics")
    elif failure == "profile_unknown":
        identity["execution_economics"]["unknown"] = 0
    elif failure == "profile_disagreement":
        identity["execution_economics"]["spread_rate"] = 0.02
    elif failure == "units":
        dataset = replace(
            dataset,
            volume_units=(VolumeUnit.BASE_ASSET,) * 2,
            identity_payload_json=None,
        )
    elif failure in {
        "cash",
        "mark",
        "funding_due",
        "funding_count",
        "funding_rate_without_events",
    }:
        field = {
            "cash": "cash_rate",
            "mark": "mark_price",
            "funding_count": "funding_event_count",
            "funding_due": "funding_due",
            "funding_rate_without_events": "funding_rate",
        }[failure]
        values = dataset.resolved_array(field).copy()
        if failure == "funding_count":
            supplied = recipe(funding_minimum_event_count=4)
        elif failure == "cash":
            values[24] = 0.01
            dataset = replace(dataset, cash_rate=values, identity_payload_json=None)
        else:
            values[24 if failure in {"cash", "mark"} else 8, 0] += (
                0.01 if failure in {"cash", "mark"} else 1
            )
            if failure == "funding_due":
                values[8, 0] = False
            if failure == "funding_rate_without_events":
                values[9, 0] = 0.0001
            dataset = replace(dataset, **{field: values}, identity_payload_json=None)
    elif failure == "history":
        supplied = recipe(funding_lookback_hours=25)
    elif failure == "coverage":
        supplied = recipe(funding_coverage_start=time(1))
    elif failure == "future_source":
        supplied = recipe(assumptions_available_at=time(25))
    elif failure == "unavailable_interval":
        available = dataset.resolved_array("available_at").copy()
        available[7, 0] = time(25)
        information = dataset.resolved_array("information_available").copy()
        information[7, 0] = False
        dataset = replace(
            dataset,
            available_at=available,
            information_available=information,
            identity_payload_json=None,
        )
    elif failure == "gap":
        supplied = recipe(funding_maximum_counted_bar_close_gap_hours=7)
    elif failure == "age":
        supplied = recipe(funding_maximum_counted_bar_close_age_hours=7)
    dataset = dataset.with_content_identity(identity)
    with pytest.raises(ValueError):
        estimate(dataset, supplied)


def test_non_aligned_24h_lookback_on_5h_bars_does_not_admit_partial_interval():
    dataset = market()
    identity = parse_identity_json(dataset.identity_payload_json)
    timestamps = time(0) + np.arange(72) * np.timedelta64(5, "h")
    dataset = replace(
        dataset,
        timestamps=timestamps,
        available_at=np.broadcast_to(timestamps[:, None], (72, 2)),
        nominal_bar_hours=5.0,
        periods_per_year=1752,
        identity_payload_json=None,
    ).with_content_identity(identity)
    with pytest.raises(ValueError, match="align"):
        estimate(dataset, recipe(), decisions=(24,))


@pytest.mark.parametrize(
    "decisions,horizon",
    (((24, 24), 48), ((), 48), ((True,), 48), ((72,), 48), ((24,), True)),
)
def test_invalid_decision_roster_and_horizon(decisions, horizon):
    with pytest.raises(ValueError):
        estimate(decisions=decisions, horizon=horizon)
