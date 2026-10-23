from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

from tests.tools.market_b4_oracle._native_replay_fixture import (
    _native_actor_identity_hashes,
    _public_native_fixture,
)
from tools.market_b4_oracle import (
    decision_identity,
)
from tools.market_b4_oracle import (
    frozen_market_replay as replay_module,
)
from tools.market_b4_oracle.expected_intent import ExpectedIntent
from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
from tools.market_b4_oracle.frozen_market_replay import (
    FrozenMarketBinding,
    FrozenMarketPolicy,
    FrozenMarketReplay,
    FrozenMarketWindow,
    ObservedFill,
    _apply_owned_fill_progress,
    _OwnedOrder,
    _transition_order,
)
from tools.market_replay_oracle.artifact import VerifiedDatasetContent

DATASET_ID = "a" * 64
POLICY_DIGEST = "b" * 64
RESERVE_PLAN_DIGEST = "c" * 64
BASE_NS = 1_768_000_000_000_000_000
HOUR_NS = 3_600_000_000_000
FEATURE_NAMES = tuple(f"feature_{index}" for index in range(6))
EXECUTION_ECONOMICS = {
    "schema_version": "execution_economics_profile_v1",
    "name": "frozen market replay synthetic profile",
    "fee_rate": 0.0005,
    "maker_fee_rate": 0.0,
    "taker_fee_rate": 0.0,
    "spread_rate": 0.0002,
    "max_participation_rate": 0.05,
    "borrow_available": True,
    "borrow_rate": 0.0,
}


def _content(*, rows: int = 65) -> VerifiedDatasetContent:
    timestamps = np.asarray(
        [BASE_NS + index * HOUR_NS for index in range(rows)], dtype="datetime64[ns]"
    )
    prices = np.full((rows, 2), 100.0, dtype=np.float64)
    features = np.zeros((rows, 2, 6), dtype=np.float64)
    for row in range(rows):
        features[row, :, :] = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    arrays: dict[str, np.ndarray] = {
        "timestamps": timestamps,
        "available_at": np.broadcast_to(timestamps[:, None], (rows, 2)).copy(),
        "information_available": np.ones((rows, 2), dtype=np.bool_),
        "features": features,
        "feature_available": np.ones((rows, 2, 6), dtype=np.bool_),
        "open": prices.copy(),
        "high": prices.copy(),
        "low": prices.copy(),
        "close": prices.copy(),
        "volume": np.full((rows, 2), 3_000_000.0, dtype=np.float64),
        "funding_rate": np.zeros((rows, 2), dtype=np.float64),
        "funding_price_rate": np.zeros((rows, 2), dtype=np.float64),
        "funding_event_count": np.zeros((rows, 2), dtype=np.int32),
        "tradable": np.ones((rows, 2), dtype=np.bool_),
        "symbol_active": np.ones((rows, 2), dtype=np.bool_),
        "asset_active": np.ones((rows, 2), dtype=np.bool_),
        "fee_rate": np.full(
            (rows, 2), EXECUTION_ECONOMICS["fee_rate"], dtype=np.float64
        ),
        "maker_fee_rate": np.zeros((rows, 2), dtype=np.float64),
        "taker_fee_rate": np.full(
            (rows, 2), EXECUTION_ECONOMICS["taker_fee_rate"], dtype=np.float64
        ),
        "spread_rate": np.full(
            (rows, 2), EXECUTION_ECONOMICS["spread_rate"], dtype=np.float64
        ),
        "max_participation_rate": np.full(
            (rows, 2), EXECUTION_ECONOMICS["max_participation_rate"], dtype=np.float64
        ),
        "minimum_notional": np.zeros((rows, 2), dtype=np.float64),
        "lot_size": np.full((rows, 2), 0.001, dtype=np.float64),
        "tick_size": np.full((rows, 2), 0.1, dtype=np.float64),
        "borrow_available": np.ones((rows, 2), dtype=np.bool_),
        "borrow_rate": np.full(
            (rows, 2), EXECUTION_ECONOMICS["borrow_rate"], dtype=np.float64
        ),
        "funding_due": np.zeros((rows, 2), dtype=np.bool_),
        "buy_allowed": np.ones((rows, 2), dtype=np.bool_),
        "sell_allowed": np.ones((rows, 2), dtype=np.bool_),
        "mark_price": prices.copy(),
        "index_price": prices.copy(),
        "dividend": np.zeros((rows, 2), dtype=np.float64),
        "split_factor": np.ones((rows, 2), dtype=np.float64),
        "delisting_recovery": np.ones((rows, 2), dtype=np.float64),
        "cash_rate": np.zeros(rows, dtype=np.float64),
        "contract_multipliers": np.ones(2, dtype=np.float64),
    }
    for row, rate in (
        (8, 0.0001),
        (16, 0.0005),
        (24, 0.0002),
        (32, 0.0001),
        (40, 0.0005),
        (48, 0.0002),
        (56, 0.0001),
        (64, 0.0005),
    ):
        if row < rows:
            arrays["funding_rate"][row, :] = rate
            arrays["funding_event_count"][row, :] = 1
            arrays["funding_due"][row, :] = True
    for value in arrays.values():
        value.setflags(write=False)
    dataset = MappingProxyType(
        {
            "symbols": ("BTC", "ETH"),
            "feature_names": FEATURE_NAMES,
            "global_feature_names": (),
            "volume_units": ("quote_notional", "quote_notional"),
            "calendar_kind": "continuous_24_7",
            "nominal_bar_hours": 1.0,
            "periods_per_year": 8760,
            "identity_payload_json": json.dumps(
                {
                    "schema": "market_dataset_identity_v7",
                    "execution_economics": EXECUTION_ECONOMICS,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    )
    return VerifiedDatasetContent(DATASET_ID, MappingProxyType(arrays), dataset)


def _forecast_index(*, through: int = 32, expected_return: float = 0.02):
    times = [BASE_NS + index * HOUR_NS for index in range(through)]
    vintage_digest = "d" * 64
    cutoff = times[0] - 2 * HOUR_NS
    vintage = {
        "schema": "simple_return_ridge_vintage_v1",
        "return_unit": "expected_simple_return",
        "forecast_kind": "regularized_linear_projection",
        "valuation_basis": "same_close_price_return",
        "calibration_state": "uncalibrated",
        "variance_kind": "fit_prefix_marginal_variance",
        "block": {
            "fit_cutoff": cutoff,
            "model_fit_time": cutoff,
            "prediction_start": times[0],
            "prediction_stop": times[-1] + HOUR_NS,
            "inference_delay_seconds": 0,
        },
        "training": {
            "schema": "simple_return_training_scope_v1",
            "log_training": {
                "schema": "forecast_training_scope_v1",
                "feature_indices": [0, 1],
                "feature_names": [FEATURE_NAMES[0], FEATURE_NAMES[1]],
                "features": [[0.1, 0.2]],
                "labels": [0.01],
                "sample_weights": [1.0],
                "horizon_hours": 1,
                "fit_cutoff": cutoff,
                "trace": {},
            },
            "labels": [0.01],
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": 0.01,
        },
        "model": {
            "feature_indices": [0, 1],
            "feature_mean": [0.0, 0.0],
            "feature_scale": [1.0, 1.0],
            "coefficients": [0.0, 0.0],
            "intercept": 0.0,
            "horizon_hours": 1,
            "alpha": 1.0,
            "n_samples": 1,
            "fit_cutoff": cutoff,
            "return_unit": "expected_simple_return",
            "valuation_basis": "same_close_price_return",
            "variance_kind": "fit_prefix_marginal_variance",
            "fit_prefix_marginal_variance": 0.01,
        },
        "prediction_symbols": ["BTC", "ETH"],
        "digest": vintage_digest,
    }
    packets = []
    for index, timestamp in enumerate(times):
        for symbol in ("BTC", "ETH"):
            packets.append(
                {
                    "schema": "simple_return_packet_v1",
                    "symbol": symbol,
                    "as_of": timestamp,
                    "source_available_at": timestamp,
                    "feature_values": [0.1, 0.2],
                    "decision_close": 100.0,
                    "snapshot_digest": "e" * 64,
                    "forecast_available_at": timestamp,
                    "horizon_end": timestamp + HOUR_NS,
                    "horizon_seconds": 3600,
                    "expected_simple_return": expected_return,
                    "fit_prefix_marginal_variance": 0.01,
                    "vintage_digest": vintage_digest,
                    "return_unit": "expected_simple_return",
                    "valuation_basis": "same_close_price_return",
                    "variance_kind": "fit_prefix_marginal_variance",
                    "forecast_kind": "regularized_linear_projection",
                    "digest": hashlib.sha256(
                        f"packet-{index}-{symbol}".encode()
                    ).hexdigest(),
                }
            )
    stream = {
        "schema": "simple_return_ridge_stream_v1",
        "availability_semantics": "declared_simulation_v1",
        "dataset_id": DATASET_ID,
        "vintages": [vintage],
        "packets": packets,
        "causal_scope_digest": "1" * 64,
    }
    raw = json.dumps(stream, allow_nan=False, separators=(",", ":")).encode()
    return FrozenForecastIndex(
        raw,
        pinned_bytes=len(raw),
        pinned_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _policy(*, latency: int = 0, multiplier: float = 2.0) -> FrozenMarketPolicy:
    return FrozenMarketPolicy(
        dataset_id=DATASET_ID,
        execution_policy_digest=POLICY_DIGEST,
        order_latency_bars=latency,
        fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0001,
        multiplier=multiplier,
        max_participation_rate=0.05,
        allow_short=True,
        max_leverage=1.0,
        maintenance_margin_rate=0.25,
        collateral_haircut=1.0,
        borrow_rate_multiplier=1.0,
        maximum_gross=1.0,
        drawdown_start=0.1,
        drawdown_stop=0.2,
        insolvency_valuation="retain_debt",
        margin_mode="cross",
        tick_size_floor=0.0,
        lot_size_floor=0.0,
        minimum_notional_floor=0.0,
        tick_size_factor=1.0,
        lot_size_factor=1.0,
        minimum_notional_factor=1.0,
        adverse_tick_rounding=False,
    )


def _runner(
    selected_symbol: int, *, latency: int = 0, initial_capital: float = 100_000.0
):
    scenario = {0: "COST2L0", 2: "BASEL2"}.get(latency)
    if scenario is None:
        raise ValueError("synthetic replay supports only frozen profile latencies")
    content = _content()
    window = FrozenMarketWindow(actor_start=24, actor_stop=56, terminal_stop=64)
    binding = FrozenMarketBinding._comparison_fixture(
        content=content,
        window=window,
        selected_symbol_index=selected_symbol,
        expected_horizon_seconds=3600,
        observation_feature_indices=tuple(range(6)),
        native_role="nonrl",
        initial_capital=initial_capital,
        scenario=scenario,
    )
    return FrozenMarketReplay(
        content=content,
        policy=_policy(latency=latency, multiplier=1.0 if latency == 2 else 2.0),
        binding=binding,
        reserve_plan_digest=RESERVE_PLAN_DIGEST,
    )


def _alternate_account_scope(binding: FrozenMarketBinding) -> FrozenMarketBinding:
    document = binding.protocol_document
    configuration_id = f"{binding.configuration_id}_CACHE_SCOPE"
    document["configurations"][0]["id"] = configuration_id
    symbol = document["dataset"]["fit_symbols"][binding.selected_symbol_index]
    account_id = f"{configuration_id}_{symbol}"
    document["accounts"] = [account_id]
    return replace(
        binding,
        protocol_canonical=decision_identity.canonical_json_bytes(document),
        account_id=account_id,
        configuration_id=configuration_id,
    )


def test_observed_fill_cannot_drive_account_before_latency_eligibility() -> None:
    runner = _runner(0, latency=2)
    before = runner.account_snapshot

    with pytest.raises(ValueError, match="observed fill batch differs"):
        runner.advance(
            forecast_index=_forecast_index(),
            role="NONRL",
            observed_fills=(ObservedFill("c" * 64, 1, Fraction(1), 100.0),),
        )

    after = runner.account_snapshot
    assert (
        after.exact_quantities == before.exact_quantities == (Fraction(0), Fraction(0))
    )
    assert after.cash == before.cash == 100_000.0
    assert runner.current_index == 24


@pytest.mark.parametrize("selected_symbol", [0, 1])
def test_selected_symbol_owns_market_fill_and_other_slot_stays_flat(
    selected_symbol: int,
) -> None:
    runner = _runner(selected_symbol)
    result = runner.advance(forecast_index=_forecast_index(), role="NONRL")

    assert result.processing_index == 25
    assert result.decision_index == 24
    assert result.scope == "OWNED_RAW_MARKET_COMPARISON_ONLY"
    assert result.owned_fills
    assert runner.account_snapshot.exact_quantities[1 - selected_symbol] == Fraction(0)
    assert runner.active_order_count <= 1


def test_latency_waits_for_eligible_raw_bar_before_filling() -> None:
    runner = _runner(0, latency=2)
    forecast_index = _forecast_index(through=64)

    first = runner.advance(forecast_index=forecast_index, role="NONRL")
    assert first.owned_fills == ()
    assert runner.account_snapshot.exact_quantities == (Fraction(0), Fraction(0))
    assert runner.active_order_count == 1

    second = runner.advance(forecast_index=forecast_index, role="NONRL")
    assert second.processing_index == 26
    assert second.owned_fills
    assert runner.account_snapshot.exact_quantities[0] > 0


def test_partial_fill_charges_once_and_later_hold_reconciles_the_remainder() -> None:
    runner = _runner(0, initial_capital=2_000_000.0)
    forecast_index = _forecast_index(through=64)

    first = runner.advance(forecast_index=forecast_index, role="NONRL")

    assert first.owned_fills
    assert runner.account_snapshot.total_cost == pytest.approx(
        sum(fill.cost_amount for fill in first.owned_fills)
    )
    assert runner.active_order_count == 1
    assert abs(runner.account_snapshot.exact_quantities[0]) == Fraction(1500)

    hold = runner.advance(
        forecast_index=_forecast_index(through=64, expected_return=0.0),
        role="NONRL",
    )

    assert hold.cancelled_order_ids == (first.new_intent_ids[0],)
    assert runner.account_snapshot.exact_quantities == (Fraction(0), Fraction(0))
    assert runner.active_order_count <= 1


def test_terminal_order_intent_is_not_rebuilt_after_its_archive_is_warm(
    monkeypatch,
) -> None:
    runner = _runner(0)
    forecast_index = _forecast_index(through=64)

    first = runner.advance(forecast_index=forecast_index, role="NONRL")
    assert first.owned_fills
    assert runner._terminal
    terminal_order_id, terminal_record = next(iter(runner._terminal.items()))

    calls_for_terminal_order = 0
    original_create = ExpectedIntent.create

    def observe_create(cls, **kwargs):
        nonlocal calls_for_terminal_order
        intent = original_create(**kwargs)
        if intent.order_id == terminal_order_id:
            calls_for_terminal_order += 1
        return intent

    monkeypatch.setattr(ExpectedIntent, "create", classmethod(observe_create))

    # The first later actor row builds the identity with this now-terminal order.
    runner.advance(forecast_index=forecast_index, role="NONRL")
    warm_conversion_count = calls_for_terminal_order
    assert warm_conversion_count > 0

    for _ in range(2):
        runner.advance(forecast_index=forecast_index, role="NONRL")
        assert runner._terminal[terminal_order_id] is terminal_record

    assert calls_for_terminal_order == warm_conversion_count


def test_projection_memo_does_not_hide_distinct_source_with_same_order_id() -> None:
    runner = _runner(0)
    runner.advance(forecast_index=_forecast_index(through=64), role="NONRL")
    order_id, source = next(iter(runner._terminal.items()))
    projection_memo = {}

    original = runner._owned_projection(
        source, runner._active, runner._terminal, projection_memo
    )
    changed_source = replace(source, evidence_version=source.evidence_version + 1)
    runner._terminal[order_id] = changed_source
    runner._all_orders[order_id] = changed_source
    changed = runner._owned_projection(
        changed_source, runner._active, runner._terminal, projection_memo
    )

    assert changed is not original
    assert changed.evidence_version == source.evidence_version + 1

    malformed_source = replace(changed_source, actor_identity=None)
    runner._terminal[order_id] = malformed_source
    runner._all_orders[order_id] = malformed_source
    with pytest.raises(
        ValueError, match="reserve identity cannot be projected as an actor identity"
    ):
        runner._owned_projection(
            malformed_source, runner._active, runner._terminal, projection_memo
        )


@pytest.mark.parametrize("invalidation", ["scope", "terminal_owner"])
def test_terminal_identity_memo_rebuilds_after_scope_or_owner_change(
    invalidation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(0)
    forecast_index = _forecast_index(through=64)
    runner.advance(forecast_index=forecast_index, role="NONRL")
    terminal_order_id, original_source = next(iter(runner._terminal.items()))

    original_create = ExpectedIntent.create
    warm_conversions = 0

    def observe_warm_create(cls, **kwargs):
        nonlocal warm_conversions
        intent = original_create(**kwargs)
        if intent.order_id == terminal_order_id:
            warm_conversions += 1
        return intent

    monkeypatch.setattr(ExpectedIntent, "create", classmethod(observe_warm_create))
    runner.advance(forecast_index=forecast_index, role="NONRL")
    assert warm_conversions > 0
    warm_conversion_count = warm_conversions
    runner.advance(forecast_index=forecast_index, role="NONRL")
    assert warm_conversions == warm_conversion_count

    if invalidation == "scope":
        runner.binding = _alternate_account_scope(runner.binding)
    else:
        changed = replace(
            original_source, evidence_version=original_source.evidence_version + 1
        )
        runner._terminal[terminal_order_id] = changed
        runner._all_orders[terminal_order_id] = changed

    core_rebuilds = 0
    inside_core = False

    def observe_create(cls, **kwargs):
        nonlocal core_rebuilds
        intent = original_create(**kwargs)
        if inside_core and intent.order_id == terminal_order_id:
            core_rebuilds += 1
        return intent

    monkeypatch.setattr(ExpectedIntent, "create", classmethod(observe_create))
    original_core = replay_module._build_actor_identity_core
    compared = 0

    def compare_with_public_builder(*args, **kwargs):
        nonlocal compared, inside_core
        inside_core = True
        try:
            identity = original_core(*args, **kwargs)
        finally:
            inside_core = False
        reference_kwargs = {
            key: value for key, value in kwargs.items() if key != "terminal_memo"
        }
        reference = decision_identity.build_actor_identity(*args, **reference_kwargs)
        assert identity == reference
        compared += 1
        return identity

    monkeypatch.setattr(
        replay_module, "_build_actor_identity_core", compare_with_public_builder
    )
    runner.advance(forecast_index=forecast_index, role="NONRL")

    assert compared == 1
    assert core_rebuilds > 0


def test_mutable_terminal_progress_alias_stays_live_in_replay_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(0)
    forecast_index = _forecast_index(through=64)
    runner.advance(forecast_index=forecast_index, role="NONRL")
    terminal_order_id, source = next(iter(runner._terminal.items()))
    assert type(source.last_processed_index) is int
    alias = [source.last_processed_index]
    aliased_source = replace(source, last_processed_index=alias)
    runner._terminal[terminal_order_id] = aliased_source
    runner._all_orders[terminal_order_id] = aliased_source

    original_create = ExpectedIntent.create
    core_rebuilds = 0
    inside_core = False

    def observe_create(cls, **kwargs):
        nonlocal core_rebuilds
        intent = original_create(**kwargs)
        if inside_core and intent.order_id == terminal_order_id:
            core_rebuilds += 1
        return intent

    monkeypatch.setattr(ExpectedIntent, "create", classmethod(observe_create))
    original_core = replay_module._build_actor_identity_core
    compared = 0
    identities = []

    def compare_with_public_builder(*args, **kwargs):
        nonlocal compared, inside_core
        inside_core = True
        try:
            identity = original_core(*args, **kwargs)
        finally:
            inside_core = False
        reference_kwargs = {
            key: value for key, value in kwargs.items() if key != "terminal_memo"
        }
        reference = decision_identity.build_actor_identity(*args, **reference_kwargs)
        assert identity == reference
        compared += 1
        identities.append(identity)
        return identity

    monkeypatch.setattr(
        replay_module, "_build_actor_identity_core", compare_with_public_builder
    )
    first = runner.advance(forecast_index=forecast_index, role="NONRL")
    assert first.actor_context_bytes is not None
    alias.append(9_999)
    second = runner.advance(forecast_index=forecast_index, role="NONRL")
    assert second.actor_context_bytes is not None

    first_identity, second_identity = identities
    first_state = json.loads(first_identity.context_state_bytes)
    second_state = json.loads(second_identity.context_state_bytes)
    first_rows = first_state["terminal_orders"]
    second_rows = second_state["terminal_orders"]
    first_row = next(
        row for row in first_rows if row["intent"]["order_id"] == terminal_order_id
    )
    second_row = next(
        row for row in second_rows if row["intent"]["order_id"] == terminal_order_id
    )
    assert first_row["last_processed_index"] == [source.last_processed_index]
    assert second_row["last_processed_index"] == alias
    assert first_identity.context_state_bytes != second_identity.context_state_bytes
    assert first_identity.context_state_digest != second_identity.context_state_digest
    assert compared == 2
    assert core_rebuilds == 2


@pytest.mark.parametrize(
    "capacity_constant",
    [
        "_TERMINAL_IDENTITY_CACHE_MAX_ORDER_ENTRIES",
        "_TERMINAL_IDENTITY_CACHE_MAX_MAPPING_BYTES",
    ],
)
def test_terminal_identity_cache_capacity_uses_uncached_public_parity(
    capacity_constant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay_module, capacity_constant, 0)
    runner = _runner(0)
    forecast_index = _forecast_index(through=64)
    original_core = replay_module._build_actor_identity_core
    compared = 0

    def compare_with_public_builder(*args, **kwargs):
        nonlocal compared
        identity = original_core(*args, **kwargs)
        reference_kwargs = {
            key: value for key, value in kwargs.items() if key != "terminal_memo"
        }
        reference = decision_identity.build_actor_identity(*args, **reference_kwargs)
        assert identity == reference
        compared += 1
        return identity

    monkeypatch.setattr(
        replay_module, "_build_actor_identity_core", compare_with_public_builder
    )
    first = runner.advance(forecast_index=forecast_index, role="NONRL")
    assert first.owned_fills
    terminal_order_id = next(iter(runner._terminal))

    calls_for_terminal_order = 0
    original_create = ExpectedIntent.create

    def observe_create(cls, **kwargs):
        nonlocal calls_for_terminal_order
        intent = original_create(**kwargs)
        if intent.order_id == terminal_order_id:
            calls_for_terminal_order += 1
        return intent

    monkeypatch.setattr(ExpectedIntent, "create", classmethod(observe_create))
    runner.advance(forecast_index=forecast_index, role="NONRL")

    assert calls_for_terminal_order > 0
    assert compared == 2
    assert runner._terminal_identity_memo.disabled is True


@pytest.mark.parametrize("selected_symbol", [0, 1])
def test_thirty_two_actor_and_eight_reserve_steps_flatten_selected_slot(
    selected_symbol: int,
) -> None:
    runner = _runner(selected_symbol)
    forecast_index = _forecast_index(through=64)
    for _ in range(40):
        runner.advance(forecast_index=forecast_index, role="NONRL")

    assert runner.current_index == 64
    assert runner.actor_advances == 32
    assert runner.reserve_advances == 8
    assert runner.account_snapshot.exact_quantities == (Fraction(0), Fraction(0))
    assert runner.active_order_count == 0
    assert runner.source_scope == "CALLER_PINNED_INPUTS_UNAUTHENTICATED"


def test_six_allocation_costs_come_from_the_raw_decision_window() -> None:
    runner = _runner(0)
    source = runner._decision_snapshot(24)

    assumptions, source_identity = runner._derive_cost_assumptions(
        source, decision_index=24
    )

    assert assumptions.expected_simple_return == 0.0
    assert not hasattr(runner.policy, "allocation_costs")
    assert assumptions.return_variance == 0.0
    assert assumptions.buy_cost == assumptions.sell_cost == assumptions.exit_cost
    assert assumptions.buy_cost == pytest.approx(
        2.0 * (0.0005 + 0.0002 + 0.0001 * (100_000.0 / 3_000_000.0) ** 0.5)
    )
    assert assumptions.funding_return == pytest.approx(
        (0.0001 + 0.0005 + 0.0002) / 24.0
    )
    assert assumptions.borrow_return == 0.0
    assert assumptions.cash_return == 0.0
    assert len(source_identity) == 64
    recipe = runner._cost_recipe_declarations()
    protocol_identity = hashlib.sha256(
        json.dumps(
            runner.binding.protocol_document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert recipe["assumptions_identity"] == protocol_identity
    assert recipe["execution_cost"] == runner.binding.effective_profile["cost_policy"]
    assert recipe["reference_notional"] == 100_000.0
    assert recipe["funding_lookback_hours"] == 24
    assert recipe["funding_minimum_event_count"] == 2
    assert recipe["funding_maximum_counted_bar_close_gap_hours"] == 9
    assert recipe["funding_maximum_counted_bar_close_age_hours"] == 8
    assert recipe["funding_coverage_start"] == BASE_NS
    assert recipe["funding_coverage_stop"] == BASE_NS + 64 * HOUR_NS
    raw_profile = json.loads(runner.content.dataset["identity_payload_json"])[
        "execution_economics"
    ]
    expected_source = hashlib.sha256(
        json.dumps(
            {
                "dataset_id": runner.content.dataset_id,
                "execution_economics": raw_profile,
                "recipe": recipe,
                "symbol": "BTC",
                "decision_time": BASE_NS + 24 * HOUR_NS,
                "horizon_hours": 1,
                "profile_origin": "caller_declared_simulation_v1",
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert source_identity == expected_source


def test_native_pair_binding_derives_account_horizon_and_protocol_identity(
    tmp_path,
) -> None:
    from tests.tools.market_replay_oracle.test_protocol_pair import _load_pair

    pair, *_ = _load_pair(tmp_path)
    protocol = json.loads(pair.native_canonical)
    timestamps = np.datetime64("2024-12-01T01:00:00", "ns") + np.arange(
        9504
    ) * np.timedelta64(1, "h")
    timestamps.setflags(write=False)
    features = np.zeros((9504, 2, 6), dtype=np.float64)
    features.setflags(write=False)
    content = VerifiedDatasetContent(
        protocol["dataset"]["dataset_id"],
        MappingProxyType({"timestamps": timestamps, "features": features}),
        MappingProxyType(
            {
                "symbols": ("BTCUSDT", "ETHUSDT"),
                "feature_names": tuple(protocol["dataset"]["feature_names"]),
                "calendar_kind": "continuous_24_7",
                "nominal_bar_hours": 1.0,
            }
        ),
    )

    binding = FrozenMarketBinding.from_declared_protocol_pair(
        protocol_pair=pair,
        content=content,
        configuration_id="H72_RIDGE_A10",
        scenario="COST2L0",
        native_role="nonrl",
        selected_symbol_index=1,
    )

    assert binding.account_id == "H72_RIDGE_A10_ETHUSDT"
    assert binding.expected_horizon_seconds == 72 * 3600
    assert binding.window == FrozenMarketWindow(2903, 9430, 9502)
    assert binding.assumptions_identity == (
        "742e883b405a0d7f600998b820a190f872673f1a439d56dd3b8001a6cf017393"
    )
    assert not binding.comparison_only


@pytest.mark.parametrize(
    ("scenario", "expected_multiplier", "expected_latency"),
    [("BASEL0", 1.0, 0), ("COST2L0", 2.0, 0), ("BASEL2", 1.0, 2)],
)
def test_policy_accepts_only_frozen_scenario_cost_profiles(
    scenario: str, expected_multiplier: float, expected_latency: int
) -> None:
    from tools.market_replay_oracle.profiles import effective_profile

    risk = {
        "max_gross": 1.0,
        "max_abs_weight": 1.0,
        "max_turnover": None,
        "drawdown_start": 0.1,
        "drawdown_stop": 0.2,
        "emergency_turnover_override": True,
        "fail_closed_tolerance": 1e-10,
    }
    profile = effective_profile(scenario, 32, risk)
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=DATASET_ID, profile=profile, risk=risk
    )

    assert policy.multiplier == expected_multiplier
    assert policy.order_latency_bars == expected_latency


def test_replay_rejects_role_change_outside_bound_arm() -> None:
    runner = _runner(0)

    with pytest.raises(ValueError, match="decision role differs"):
        runner.advance(forecast_index=_forecast_index(), role="CASH")

    assert runner.current_index == 24


def test_actor_cost_derivation_rejects_inconsistent_raw_funding_prefix() -> None:
    content = _content()
    arrays = dict(content.arrays)
    due = arrays["funding_due"].copy()
    due[16, 0] = False
    due.setflags(write=False)
    arrays["funding_due"] = due
    inconsistent = VerifiedDatasetContent(content.dataset_id, arrays, content.dataset)
    runner = FrozenMarketReplay(
        content=inconsistent,
        policy=_policy(),
        binding=FrozenMarketBinding._comparison_fixture(
            content=inconsistent,
            window=FrozenMarketWindow(actor_start=24, actor_stop=56, terminal_stop=64),
            selected_symbol_index=0,
            expected_horizon_seconds=3600,
            observation_feature_indices=tuple(range(6)),
            native_role="nonrl",
            initial_capital=100_000.0,
        ),
        reserve_plan_digest=RESERVE_PLAN_DIGEST,
    )

    with pytest.raises(ValueError, match="due/count/rates"):
        runner._derive_cost_assumptions(
            runner._decision_snapshot(24), decision_index=24
        )


@pytest.mark.parametrize(
    ("quantity", "position", "reduce_only", "allow_short", "borrow", "reason"),
    [
        (-1.0004, Fraction("1.0001"), True, True, True, "reduce_only_exceeds_position"),
        (-0.0018, Fraction("0.0012"), False, False, True, "shorting_disabled"),
        (-0.0008, Fraction("-0.0012"), False, True, False, "borrow_unavailable"),
    ],
)
def test_admission_uses_raw_remaining_before_lot_quantization(
    quantity: float,
    position: Fraction,
    reduce_only: bool,
    allow_short: bool,
    borrow: bool,
    reason: str,
) -> None:
    runner = _runner(0)
    runner.policy = replace(runner.policy, allow_short=allow_short)
    intent = ExpectedIntent.create(
        dataset_id=DATASET_ID,
        target_identity="f" * 64,
        execution_policy_digest=POLICY_DIGEST,
        symbol_index=0,
        requested_quantity=quantity,
        order_type="market",
        time_in_force="gtc",
        limit_price=None,
        stop_price=None,
        submit_index=24,
        eligible_index=24,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=100_000.0,
        reduce_only=reduce_only,
    )
    order = _OwnedOrder(intent, actor_identity=None)
    row = {
        "lot": 0.001,
        "minimum": 0.0,
        "multipliers": (1.0, 1.0),
        "borrow_available": (borrow, True),
    }

    assert (
        runner._admission_reason(
            order,
            row,
            100_000.0,
            (position, Fraction(0)),
            (100.0, 100.0),
        )
        == reason
    )


def test_inactive_selected_asset_row_fails_closed_before_state_mutation() -> None:
    content = _content()
    arrays = dict(content.arrays)
    inactive = arrays["asset_active"].copy()
    inactive[25, 0] = False
    inactive.setflags(write=False)
    arrays["asset_active"] = inactive
    content = VerifiedDatasetContent(
        content.dataset_id, MappingProxyType(arrays), content.dataset
    )
    runner = FrozenMarketReplay(
        content=content,
        policy=_policy(),
        binding=FrozenMarketBinding._comparison_fixture(
            content=content,
            window=FrozenMarketWindow(actor_start=24, actor_stop=56, terminal_stop=64),
            selected_symbol_index=0,
            expected_horizon_seconds=3600,
            observation_feature_indices=tuple(range(6)),
            native_role="nonrl",
            initial_capital=100_000.0,
        ),
        reserve_plan_digest=RESERVE_PLAN_DIGEST,
    )
    before = runner.account_snapshot

    with pytest.raises(ValueError, match="inactive selected asset row"):
        runner.advance(forecast_index=_forecast_index(), role="NONRL")

    assert runner.current_index == 24
    assert runner.account_snapshot == before


def test_reserve_identity_uses_plan_closing_index_without_actor_identity() -> None:
    runner = _runner(0)
    forecast_index = _forecast_index(through=64)
    for _ in range(32):
        runner.advance(forecast_index=forecast_index, role="NONRL")

    reserve = runner.advance(forecast_index=forecast_index, role="NONRL")

    assert reserve.phase == "reserve"
    assert reserve.actor_identity is None
    assert reserve.target_identity == runner._reserve_target_identity(56)


@pytest.mark.parametrize("selected_symbol", [0, 1])
def test_public_native_task18_rules_partial_fill_remainder_sources_next_identity(
    tmp_path, selected_symbol: int
) -> None:
    owned, forecast_index, native, _, _ = _public_native_fixture(
        tmp_path,
        selected_symbol,
        scenario="BASEL0",
        role="nonrl",
        horizon=72,
        quote_volume=2_000_000.0,
        task18_execution_rules=True,
        rounding_partial_case=True,
    )
    tick_size = 0.1 if selected_symbol == 0 else 0.01
    minimum_notional = 50.0 if selected_symbol == 0 else 20.0
    assert owned.content.arrays["tick_size"][128, selected_symbol] == tick_size
    assert owned.content.arrays["lot_size"][128, selected_symbol] == 0.001
    assert (
        owned.content.arrays["minimum_notional"][128, selected_symbol]
        == minimum_notional
    )
    raw_open = float(owned.content.arrays["open"][129, selected_symbol])
    assert raw_open != np.round(raw_open / tick_size) * tick_size
    native_rows = native["native_receipt"]["rows"]
    first_bar = native_rows[0]["native"]
    partial_events = tuple(
        event
        for event in first_bar["order_events"]
        if event["event_type"] == "partial_fill"
    )
    assert len(partial_events) == 1, "fixture must create a native partial fill"
    native_fill = partial_events[0]
    rounded_open = float(np.round(raw_open / tick_size) * tick_size)
    assert native_fill["path_points"][0] == raw_open
    assert native_fill["execution_price"] == rounded_open
    assert native_fill["capacity_before"] == 100_000.0
    assert native_fill["filled_notional"] <= native_fill["capacity_before"]
    assert native_fill["capacity_after"] == pytest.approx(
        native_fill["capacity_before"] - native_fill["filled_notional"]
    )
    assert Fraction(str(native_fill["filled_quantity"])) == (
        native_fill["filled_lot_count"] * Fraction(str(native_fill["lot_size"]))
    )

    native_actor_rows = native["native_receipt"]["rows"]
    native_reserve_rows = native["closing_rows"]
    native_bars = [row["native"] for row in native_actor_rows] + [
        row["native"] for row in native_reserve_rows
    ]
    assert len(native_actor_rows) == 32
    assert len(native_reserve_rows) == 8
    assert native["settlement_complete"] is True

    def take_owned_step():
        terminal_before = set(owned._terminal)
        advance = owned.advance(forecast_index=forecast_index, role="NONRL")
        snapshot = owned.account_snapshot
        active_after = dict(owned._active)
        terminal_after = {
            order_id: order
            for order_id, order in owned._terminal.items()
            if order_id not in terminal_before
        }
        return advance, snapshot, active_after, terminal_after

    first = owned.advance(forecast_index=forecast_index, role="NONRL")
    first_snapshot = owned.account_snapshot
    first_active = dict(owned._active)
    first_terminal = dict(owned._terminal)
    assert len(first.owned_fills) == 1
    assert first.owned_fills[0].order_id == native_fill["order_id"]
    assert first.owned_fills[0].exact_quantity == Fraction(
        str(native_fill["filled_quantity"])
    )
    assert first.owned_fills[0].execution_price == rounded_open
    native_order = first_bar["active_orders"][0]
    owned_order = next(iter(owned._active.values()))
    assert native_order["status"] == owned_order.status == "partially_filled"
    assert native_order["evidence_version"] == owned_order.evidence_version
    assert native_order["last_processed_index"] == owned_order.last_processed_index
    assert native_order["remaining_quantity"] == owned_order.remaining_quantity

    second_bar = native_rows[1]["native"]
    expected_identity = _native_actor_identity_hashes(second_bar)
    second = owned.advance(forecast_index=forecast_index, role="NONRL")
    second_snapshot = owned.account_snapshot
    second_active = dict(owned._active)
    second_terminal = {
        order_id: order
        for order_id, order in owned._terminal.items()
        if order_id not in first_terminal
    }
    assert second.actor_proposal_digest == expected_identity["baseline"]
    assert second.actor_context_state_digest == expected_identity["context_state"]
    assert second.actor_decision_digest == expected_identity["decision"]
    assert second.actor_action_digest == expected_identity["action"]
    assert second.actor_identity == expected_identity["selected"]
    native_active = {
        order["intent"]["order_id"]: order for order in second_bar["active_orders"]
    }
    assert set(owned._active) == set(native_active)
    for order_id, own_order in owned._active.items():
        expected_order = native_active[order_id]
        assert own_order.status == expected_order["status"]
        assert own_order.evidence_version == expected_order["evidence_version"]
        assert own_order.remaining_quantity == expected_order["remaining_quantity"]

    steps = [
        (first, first_snapshot, first_active, first_terminal),
        (second, second_snapshot, second_active, second_terminal),
    ]
    steps.extend(take_owned_step() for _ in range(38))

    def assert_native_float(actual: float, expected: float, label: str) -> None:
        assert actual == pytest.approx(expected, rel=0.0, abs=1e-10), label

    for step, (
        (advance, snapshot, active_after, terminal_after),
        native_bar,
    ) in enumerate(zip(steps, native_bars, strict=True)):
        processing_index = native_bar["book"]["as_of_index"]
        assert advance.processing_index == processing_index
        book = native_bar["book"]
        assert tuple(Fraction(value) for value in book["exact_quantities"]) == (
            snapshot.exact_quantities
        ), f"processing {processing_index}: exact quantities"
        for actual, expected, label in (
            (snapshot.cash, book["cash"], "cash"),
            (snapshot.wealth, book["equity"], "equity"),
            (snapshot.peak_value, book["peak_value"], "peak value"),
            (snapshot.max_drawdown, book["max_drawdown"], "drawdown"),
            (snapshot.total_cost, book["total_cost"], "cumulative cost"),
            (snapshot.funding_pnl, book["funding_pnl"], "funding"),
            (snapshot.borrow_cost, book["borrow_cost"], "borrow"),
            (snapshot.turnover_total, book["turnover_total"], "turnover"),
            (snapshot.margin_used, book["margin_used"], "margin used"),
            (
                snapshot.maintenance_requirement,
                book["maintenance_requirement"],
                "maintenance requirement",
            ),
            (snapshot.margin_deficit, book["margin_deficit"], "margin deficit"),
        ):
            assert_native_float(
                actual, expected, f"processing {processing_index}: {label}"
            )
        assert snapshot.fill_count == book["fill_count"]
        assert snapshot.rebalance_events == book["rebalance_events"]
        assert snapshot.insolvent is book["insolvent"]
        assert snapshot.termination_reason == book["termination_reason"]
        assert tuple(snapshot.projected_quantities) == pytest.approx(
            book["quantities"], rel=0.0, abs=1e-10
        )
        assert snapshot.marks == tuple(book["mark_prices"])
        assert snapshot.returns_history
        assert_native_float(
            snapshot.returns_history[-1],
            native_bar["execution"]["interval_net_return"],
            f"processing {processing_index}: interval return",
        )

        if step < len(native_actor_rows):
            native_identity = _native_actor_identity_hashes(
                native_actor_rows[step]["native"]
            )
            assert advance.actor_proposal_digest == native_identity["baseline"]
            assert (
                advance.actor_context_state_digest == native_identity["context_state"]
            )
            assert advance.actor_decision_digest == native_identity["decision"]
            assert advance.actor_action_digest == native_identity["action"]
            assert advance.actor_identity == native_identity["selected"]
        else:
            assert advance.phase == "reserve"
            assert advance.actor_identity is None

        native_events = native_bar["order_events"]
        native_submitted = tuple(
            event["order_id"]
            for event in native_events
            if event["event_type"] == "submitted"
            and event["processing_index"] == advance.decision_index
        )
        assert advance.new_intent_ids == native_submitted
        native_fills = tuple(
            event
            for event in native_events
            if event["event_type"] in {"filled", "partial_fill"}
        )
        assert len(advance.owned_fills) == len(native_fills)
        for own_fill, expected_fill in zip(
            advance.owned_fills, native_fills, strict=True
        ):
            assert own_fill.order_id == expected_fill["order_id"]
            assert own_fill.processing_index == expected_fill["processing_index"]
            assert own_fill.exact_quantity == Fraction(
                str(expected_fill["filled_quantity"])
            )
            assert own_fill.execution_price == expected_fill["execution_price"]
            assert_native_float(
                own_fill.filled_notional,
                expected_fill["filled_notional"],
                f"processing {processing_index}: fill notional",
            )
        expected_terminal = {
            item["order_id"]: item["reason"]
            for item in native_bar["terminal_order_reasons"]
        }
        assert {
            order_id: order.terminal_reason
            for order_id, order in terminal_after.items()
        } == expected_terminal
        native_active = {
            item["intent"]["order_id"]: item for item in native_bar["active_orders"]
        }
        assert set(active_after) == set(native_active)
        for order_id, own_order in active_after.items():
            expected_order = native_active[order_id]
            assert own_order.status == expected_order["status"]
            assert own_order.evidence_version == expected_order["evidence_version"]
            assert (
                own_order.last_processed_index == expected_order["last_processed_index"]
            )
            assert own_order.remaining_quantity == expected_order["remaining_quantity"]

    completion = owned.completion()
    assert owned.current_index == 168
    assert completion["status"] == "comparison_window_consumed"
    assert completion["terminal_flat"] is True
    assert completion["no_active_orders"] is True
    assert owned.account_snapshot.exact_quantities == (Fraction(0), Fraction(0))
    assert owned.active_order_count == 0
    assert native["final_book"]["exact_quantities"] == ["0", "0"]
    assert native["final_active_orders"] == []


@pytest.mark.parametrize("selected_symbol", [0, 1])
@pytest.mark.parametrize("horizon", [24, 72])
@pytest.mark.parametrize("role", ["nonrl", "cash"])
@pytest.mark.parametrize("scenario", ["BASEL0", "COST2L0", "BASEL2"])
def test_public_native_32_actor_8_reserve_matches_independent_owned_driver(
    tmp_path,
    selected_symbol: int,
    scenario: str,
    role: str,
    horizon: int,
) -> None:
    (
        owned,
        forecast_index,
        native,
        native_estimate,
        native_assumptions_identity,
    ) = _public_native_fixture(
        tmp_path,
        selected_symbol,
        scenario=scenario,
        role=role,
        horizon=horizon,
    )
    decision_role = "NONRL" if role == "nonrl" else "CASH"

    source = owned._decision_snapshot(128)
    raw_costs, raw_cost_identity = owned._derive_cost_assumptions(
        source, decision_index=128
    )
    owned_estimate = owned._cost_estimates_payload(source, raw_costs, raw_cost_identity)
    assert owned.binding.assumptions_identity == native_assumptions_identity
    assert owned_estimate == native_estimate

    advances = []
    snapshots = []
    active_states = []
    terminal_updates = []
    for _ in range(40):
        terminal_before = set(owned._terminal)
        advances.append(
            owned.advance(forecast_index=forecast_index, role=decision_role)
        )
        snapshots.append(owned.account_snapshot)
        active_states.append(dict(owned._active))
        terminal_updates.append(
            {
                order_id: order
                for order_id, order in owned._terminal.items()
                if order_id not in terminal_before
            }
        )

    native_rows = native["native_receipt"]["rows"]
    reserve_rows = native["closing_rows"]
    assert len(native_rows) == 32
    assert len(reserve_rows) == 8
    assert native["settlement_complete"]
    actor_has_fills = any(
        row["native"]["execution"]["fill_count"] for row in native_rows
    )
    reserve_has_fills = any(
        row["native"]["execution"]["fill_count"] for row in reserve_rows
    )
    assert actor_has_fills is (role == "nonrl")
    assert reserve_has_fills is (role == "nonrl")
    assert [row["processing_index"] for row in reserve_rows] == list(range(161, 169))
    assert len(advances) == 40
    assert owned.current_index == 168
    first_native_identity = _native_actor_identity_hashes(native_rows[0]["native"])
    assert advances[0].actor_proposal_digest == first_native_identity["baseline"]
    assert (
        advances[0].actor_context_state_digest == first_native_identity["context_state"]
    )
    assert advances[0].actor_decision_digest == first_native_identity["decision"]
    assert advances[0].actor_action_digest == first_native_identity["action"]
    assert advances[0].actor_identity == first_native_identity["selected"]
    native_bars = [row["native"] for row in native_rows] + [
        row["native"] for row in reserve_rows
    ]
    assert len(native_bars) == len(advances) == len(snapshots)

    def assert_native_float(actual: float, expected: float, label: str) -> None:
        assert actual == pytest.approx(expected, rel=0.0, abs=1e-10), label

    for step, (
        advance,
        snapshot,
        active_after,
        terminal_after,
        native_bar,
    ) in enumerate(
        zip(
            advances,
            snapshots,
            active_states,
            terminal_updates,
            native_bars,
            strict=True,
        )
    ):
        processing = native_bar["book"]["as_of_index"]
        assert advance.processing_index == processing
        book = native_bar["book"]
        assert tuple(Fraction(value) for value in book["exact_quantities"]) == (
            snapshot.exact_quantities
        ), f"processing {processing}: exact quantities"
        for actual, expected, label in (
            (snapshot.cash, book["cash"], "cash"),
            (snapshot.wealth, book["equity"], "equity"),
            (snapshot.peak_value, book["peak_value"], "peak value"),
            (snapshot.max_drawdown, book["max_drawdown"], "drawdown"),
            (snapshot.total_cost, book["total_cost"], "cumulative cost"),
            (snapshot.funding_pnl, book["funding_pnl"], "funding"),
            (snapshot.borrow_cost, book["borrow_cost"], "borrow"),
            (snapshot.turnover_total, book["turnover_total"], "turnover"),
            (snapshot.margin_used, book["margin_used"], "margin used"),
            (
                snapshot.maintenance_requirement,
                book["maintenance_requirement"],
                "maintenance requirement",
            ),
            (snapshot.margin_deficit, book["margin_deficit"], "margin deficit"),
        ):
            assert_native_float(actual, expected, f"processing {processing}: {label}")
        assert snapshot.fill_count == book["fill_count"], (
            f"processing {processing}: cumulative fills"
        )
        assert snapshot.rebalance_events == book["rebalance_events"], (
            f"processing {processing}: rebalance events"
        )
        assert snapshot.insolvent is book["insolvent"]
        assert snapshot.termination_reason == book["termination_reason"]
        assert tuple(snapshot.projected_quantities) == pytest.approx(
            book["quantities"], rel=0.0, abs=1e-10
        ), f"processing {processing}: projected quantities"
        assert snapshot.marks == tuple(book["mark_prices"]), (
            f"processing {processing}: marks"
        )
        assert snapshot.returns_history
        assert_native_float(
            snapshot.returns_history[-1],
            native_bar["execution"]["interval_net_return"],
            f"processing {processing}: interval return",
        )

        if step < len(native_rows):
            native_actor = native_rows[step]["native"]
            native_identity = _native_actor_identity_hashes(native_actor)
            assert advance.actor_proposal_digest == native_identity["baseline"], (
                f"processing {processing}: baseline proposal hash"
            )
            assert (
                advance.actor_context_state_digest == native_identity["context_state"]
            ), f"processing {processing}: owned state digest"
            assert advance.actor_decision_digest == native_identity["decision"], (
                f"processing {processing}: decision hash"
            )
            assert advance.actor_action_digest == native_identity["action"], (
                f"processing {processing}: action hash"
            )
            assert advance.actor_identity == native_identity["selected"], (
                f"processing {processing}: selected actor identity"
            )

        native_events = native_bar["order_events"]
        native_submitted = tuple(
            event["order_id"]
            for event in native_events
            if event["event_type"] == "submitted"
            and event["processing_index"] == advance.decision_index
        )
        assert advance.new_intent_ids == native_submitted, (
            f"processing {processing}: submitted order identities"
        )
        native_fills = tuple(
            event
            for event in native_events
            if event["event_type"] in {"filled", "partial_fill"}
        )
        assert len(advance.owned_fills) == len(native_fills), (
            f"processing {processing}: fill event count"
        )
        for owned_fill, native_fill in zip(
            advance.owned_fills, native_fills, strict=True
        ):
            assert owned_fill.order_id == native_fill["order_id"]
            assert owned_fill.processing_index == native_fill["processing_index"]
            assert owned_fill.exact_quantity == Fraction(
                str(native_fill["filled_quantity"])
            )
            assert owned_fill.execution_price == native_fill["execution_price"]
            assert_native_float(
                owned_fill.filled_notional,
                native_fill["filled_notional"],
                f"processing {processing}: fill notional",
            )
        native_terminal = {
            item["order_id"]: item["reason"]
            for item in native_bar["terminal_order_reasons"]
        }
        owned_terminal = {
            order_id: order.terminal_reason
            for order_id, order in terminal_after.items()
        }
        assert owned_terminal == native_terminal, (
            f"processing {processing}: terminal order statuses"
        )
        native_active = {
            item["intent"]["order_id"]: item for item in native_bar["active_orders"]
        }
        assert set(active_after) == set(native_active), (
            f"processing {processing}: active order identities"
        )
        for order_id, own_order in active_after.items():
            native_order = native_active[order_id]
            assert own_order.status == native_order["status"]
            assert own_order.evidence_version == native_order["evidence_version"]
            assert (
                own_order.last_processed_index == native_order["last_processed_index"]
            )
            assert own_order.remaining_quantity == native_order["remaining_quantity"]

    assert owned.account_snapshot.exact_quantities == (Fraction(0), Fraction(0))
    assert owned.active_order_count == 0


def test_no_lot_decimal_adjacent_ulp_fill_matches_native_order_completion() -> None:
    from trade_rl.simulation.orders.model import (
        OrderIntent,
        OrderStatus,
        OrderType,
        PendingOrder,
        TimeInForce,
    )

    requested = 0.3
    fill_quantity = math.nextafter(requested, 0.0)
    native_intent = OrderIntent.create(
        dataset_id=DATASET_ID,
        target_identity="c" * 64,
        execution_policy_digest=POLICY_DIGEST,
        symbol_index=0,
        requested_quantity=requested,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=0,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    expected_intent = ExpectedIntent.create(
        dataset_id=DATASET_ID,
        target_identity="c" * 64,
        execution_policy_digest=POLICY_DIGEST,
        symbol_index=0,
        requested_quantity=requested,
        order_type="market",
        time_in_force="gtc",
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=0,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    native = PendingOrder.from_intent(native_intent).mark_eligible(processing_index=0)
    owned = _transition_order(
        _OwnedOrder(expected_intent, actor_identity=None),
        status="eligible",
        last_processed_index=0,
    )

    notional = fill_quantity * 100.0
    native = native.apply_fill(
        quantity=fill_quantity,
        notional=notional,
        processing_index=0,
    )
    owned = _apply_owned_fill_progress(
        owned,
        exact_fill=Fraction(str(fill_quantity)),
        filled_notional=notional,
        processing_index=0,
    )

    assert native.status is OrderStatus.FILLED
    assert native.remaining_quantity == 0.0
    assert native.cumulative_filled_quantity == requested
    assert native.exact_cumulative_filled_quantity == str(Fraction("0.3"))
    assert native.evidence_version == 2
    assert owned.status == native.status.value
    assert owned.remaining_exact == Fraction(0)
    assert owned.remaining_quantity == native.remaining_quantity
    assert owned.exact_filled_quantity == Fraction(
        native.exact_cumulative_filled_quantity
    )
    assert owned.evidence_version == native.evidence_version
