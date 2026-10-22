from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import tools.market_b4_oracle.reference_ledger as reference_ledger
from tests.tools.market_b4_oracle.test_frozen_market_replay import (
    _forecast_index,
    _runner,
)
from tools.market_account_oracle import assert_observed_account
from tools.market_b4_oracle import offline_replay_consumer as consumer
from tools.market_b4_oracle.frozen_market_replay import AccountSnapshot, ReplayAdvance
from tools.market_replay_oracle import VerifiedDatasetContent


def _ledger_for(replay):
    return consumer._NativeReferenceLedger(
        content=replay.content,
        policy=replay.policy,
        binding=replay.binding,
    )


def _observed_account(snapshot: AccountSnapshot) -> dict[str, object]:
    return {
        "cash": snapshot.cash,
        "wealth": snapshot.wealth,
        "exact_quantities": tuple(str(value) for value in snapshot.exact_quantities),
        "total_cost": snapshot.total_cost,
        "funding_pnl": snapshot.funding_pnl,
        "borrow_cost": snapshot.borrow_cost,
        "cash_interest": snapshot.cash_interest,
        "dividend": snapshot.dividend,
        "settlement_proceeds": snapshot.settlement_proceeds,
        "max_native_drawdown": snapshot.max_drawdown,
    }


def test_synthetic_reference_ledger_tracks_actor_and_reserve_accounting() -> None:
    replay = _runner(0)
    ledger = _ledger_for(replay)
    with pytest.raises(
        ValueError, match="independent reference ledger lacks complete row coverage"
    ):
        ledger.assert_complete()
    forecast = _forecast_index(through=64)
    phases: set[str] = set()
    fill_count = 0
    actor_boundary = None

    for _ in range(replay.window.terminal_stop - replay.current_index):
        advance = replay.advance(forecast_index=forecast, role="NONRL")
        phases.add(advance.phase)
        fill_count += len(advance.owned_fills)
        if (
            advance.phase == "reserve"
            and advance.decision_index == replay.window.actor_stop
        ):
            assert actor_boundary is not None
            assert ledger.account.snapshot() == actor_boundary
        ledger.apply_advance(
            advance=advance,
            observed_fills=tuple(
                fill.observed_projection() for fill in advance.owned_fills
            ),
            observed_account=replay.account_snapshot,
        )
        if advance.processing_index == replay.window.actor_stop:
            actor_boundary = ledger.account.snapshot()

    assert phases == {"actor", "reserve"}
    assert fill_count > 0
    assert replay.current_index == replay.window.terminal_stop
    ledger.assert_complete()
    assert (
        ledger.account.snapshot().exact_quantities
        == replay.account_snapshot.exact_quantities
    )
    assert_observed_account(
        ledger.account.snapshot(), _observed_account(replay.account_snapshot)
    )
    assert ledger.account.snapshot().cash_interest == 0
    assert ledger.account.snapshot().borrow_cost == 0


def test_synthetic_reference_ledger_rederives_cost_from_dataset_and_policy() -> None:
    replay = _runner(0)
    ledger = _ledger_for(replay)
    advance = replay.advance(
        forecast_index=_forecast_index(through=64),
        role="NONRL",
    )
    assert advance.owned_fills
    # This private arithmetic fixture feeds the independent raw-fill projection.
    # A deliberately corrupted Q-side cost field must not become its reference.
    changed_owned_fills = tuple(
        replace(fill, cost_amount=fill.cost_amount + 123.0)
        for fill in advance.owned_fills
    )
    changed_advance = replace(advance, owned_fills=changed_owned_fills)

    ledger.apply_advance(
        advance=changed_advance,
        observed_fills=tuple(
            fill.observed_projection() for fill in advance.owned_fills
        ),
        observed_account=replay.account_snapshot,
    )
    assert ledger.account.snapshot().total_cost > 0
    assert ledger.account.snapshot().cash < replay.initial_capital


def test_synthetic_reference_ledger_tracks_nonunit_multiplier_and_raw_carry() -> None:
    base = _runner(0)
    arrays = {
        name: np.array(values, copy=True)
        for name, values in base.content.arrays.items()
    }
    arrays["contract_multipliers"][:] = (2.0, 1.0)
    arrays["funding_price_rate"][32, 0] = 0.0001
    arrays["dividend"][40, 0] = 0.25
    for values in arrays.values():
        values.setflags(write=False)
    content = VerifiedDatasetContent(
        base.content.dataset_id,
        MappingProxyType(arrays),
        base.content.dataset,
    )
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay

    replay = FrozenMarketReplay(
        content=content,
        policy=base.policy,
        binding=base.binding,
        reserve_plan_digest=base.reserve_plan_digest,
    )
    ledger = _ledger_for(replay)
    forecast = _forecast_index(through=64)
    for _ in range(replay.window.terminal_stop - replay.current_index):
        advance = replay.advance(forecast_index=forecast, role="NONRL")
        ledger.apply_advance(
            advance=advance,
            observed_fills=tuple(
                fill.observed_projection() for fill in advance.owned_fills
            ),
            observed_account=replay.account_snapshot,
        )

    assert ledger.account._multipliers == (Decimal("2.0"), Decimal("1.0"))
    assert ledger.account.snapshot().total_cost > 0
    assert ledger.account.snapshot().funding_pnl != 0
    assert ledger.account.snapshot().dividend == Decimal("250.000000000")
    assert ledger.account.snapshot().cash_interest == 0
    assert ledger.account.snapshot().borrow_cost == 0
    assert_observed_account(
        ledger.account.snapshot(), _observed_account(replay.account_snapshot)
    )


@pytest.mark.parametrize("volume_unit", ["base_asset", "contracts"])
def test_fixed_reference_profile_rejects_unsupported_volume_units(
    volume_unit: str,
) -> None:
    base = _runner(0)
    dataset = dict(base.content.dataset)
    dataset["volume_units"] = (volume_unit, "quote_notional")
    content = VerifiedDatasetContent(
        base.content.dataset_id,
        base.content.arrays,
        MappingProxyType(dataset),
    )

    with pytest.raises(ValueError, match="requires quote-notional raw volume"):
        consumer._NativeReferenceLedger(
            content=content,
            policy=base.policy,
            binding=base.binding,
        )


def test_reference_market_price_rounds_adversely_for_each_fill_direction() -> None:
    from tools.market_account_oracle import MarketRow, market_order_facts

    previous = MarketRow(0, 100.0, 100.0, 1000.0)
    current = MarketRow(1, 100.04, 100.04, 1000.0)
    buy = market_order_facts(
        current,
        previous=previous,
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.1,
        direction=1,
        adverse_tick_rounding=True,
    )
    sell = market_order_facts(
        current,
        previous=previous,
        volume_unit="quote_notional",
        contract_multiplier=1.0,
        tick_size=0.1,
        direction=-1,
        adverse_tick_rounding=True,
    )

    assert float(buy.execution_price) == pytest.approx(100.1)
    assert float(sell.execution_price) == pytest.approx(100.0)


def test_synthetic_replay_supplies_signed_fill_direction_to_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _runner(0)
    ledger = _ledger_for(replay)
    forecast = _forecast_index(through=64)
    directions: list[int] = []
    actual = reference_ledger.market_order_facts

    def record_direction(*args: object, **kwargs: object):
        direction = kwargs.get("direction")
        assert type(direction) is int
        directions.append(direction)
        return actual(*args, **kwargs)

    monkeypatch.setattr(reference_ledger, "market_order_facts", record_direction)
    for _ in range(replay.window.terminal_stop - replay.current_index):
        advance = replay.advance(forecast_index=forecast, role="NONRL")
        ledger.apply_advance(
            advance=advance,
            observed_fills=tuple(
                fill.observed_projection() for fill in advance.owned_fills
            ),
            observed_account=replay.account_snapshot,
        )

    assert 1 in directions
    assert -1 in directions


def test_reference_account_does_not_commit_a_mismatched_fill() -> None:
    replay = _runner(0)
    ledger = _ledger_for(replay)
    forecast = _forecast_index(through=64)

    for _ in range(replay.window.terminal_stop - replay.current_index):
        advance = replay.advance(forecast_index=forecast, role="NONRL")
        observed_fills = tuple(
            fill.observed_projection() for fill in advance.owned_fills
        )
        if observed_fills:
            before = ledger.account.snapshot()
            wrong = replace(
                observed_fills[0],
                execution_price=observed_fills[0].execution_price + 0.01,
            )
            with pytest.raises(ValueError, match="native fill price differs"):
                ledger.apply_advance(
                    advance=advance,
                    observed_fills=(wrong, *observed_fills[1:]),
                    observed_account=replay.account_snapshot,
                )
            assert ledger.account.snapshot() == before
            ledger.apply_advance(
                advance=advance,
                observed_fills=observed_fills,
                observed_account=replay.account_snapshot,
            )
            break
        ledger.apply_advance(
            advance=advance,
            observed_fills=observed_fills,
            observed_account=replay.account_snapshot,
        )
    else:
        pytest.fail("synthetic replay produced no fill for mismatch coverage")


def test_reference_ledger_accepts_q_sell_at_nonaligned_nominal_tick() -> None:
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay

    first = _runner(0)
    first_fill_index = None
    forecast = _forecast_index(through=64)
    for _ in range(first.window.terminal_stop - first.current_index):
        advance = first.advance(forecast_index=forecast, role="NONRL")
        if any(fill.exact_quantity < 0 for fill in advance.owned_fills):
            first_fill_index = advance.processing_index
            break
    assert first_fill_index is not None

    arrays = {
        name: np.array(values, copy=True)
        for name, values in first.content.arrays.items()
    }
    arrays["open"][first_fill_index, 0] = 100.04
    arrays["high"][first_fill_index, 0] = 100.04
    arrays["low"][first_fill_index, 0] = 100.0
    for values in arrays.values():
        values.setflags(write=False)
    content = VerifiedDatasetContent(
        first.content.dataset_id,
        MappingProxyType(arrays),
        first.content.dataset,
    )
    replay = FrozenMarketReplay(
        content=content,
        policy=first.policy,
        binding=first.binding,
        reserve_plan_digest=first.reserve_plan_digest,
    )
    ledger = _ledger_for(replay)
    negative_fill: ReplayAdvance | None = None
    for _ in range(replay.window.terminal_stop - replay.current_index):
        advance = replay.advance(forecast_index=forecast, role="NONRL")
        ledger.apply_advance(
            advance=advance,
            observed_fills=tuple(
                fill.observed_projection() for fill in advance.owned_fills
            ),
            observed_account=replay.account_snapshot,
        )
        if any(fill.exact_quantity < 0 for fill in advance.owned_fills):
            negative_fill = advance
            break

    assert negative_fill is not None
    assert negative_fill.processing_index == first_fill_index
    assert negative_fill.owned_fills[0].execution_price == 100.0
    assert_observed_account(
        ledger.account.snapshot(), _observed_account(replay.account_snapshot)
    )


def _complete_native_content(exact_fill, binding) -> VerifiedDatasetContent:
    base = _runner(0).content
    timestamps = exact_fill.content.arrays["timestamps"]
    row_count = timestamps.size
    source_rows = np.arange(row_count) % base.arrays["timestamps"].size
    arrays = {
        name: np.take(values, source_rows, axis=0)
        if values.ndim and values.shape[0] == base.arrays["timestamps"].size
        else np.array(values, copy=True)
        for name, values in base.arrays.items()
    }
    arrays["timestamps"] = np.array(timestamps, copy=True)
    arrays["lot_size"] = np.array(exact_fill.content.arrays["lot_size"], copy=True)
    arrays["available_at"] = np.broadcast_to(timestamps[:, None], (row_count, 2)).copy()
    for values in arrays.values():
        values.setflags(write=False)
    dataset = dict(base.dataset)
    protocol_dataset = binding.protocol_document["dataset"]
    dataset.update(
        symbols=tuple(protocol_dataset["fit_symbols"]),
        feature_names=tuple(protocol_dataset["feature_names"]),
        volume_units=("quote_notional", "quote_notional"),
    )
    return VerifiedDatasetContent(
        exact_fill.content.dataset_id,
        MappingProxyType(arrays),
        MappingProxyType(dataset),
    )


def _native_forecast_bytes(content: VerifiedDatasetContent, binding) -> bytes:
    protocol = binding.protocol_document
    arrays = content.arrays
    timestamps = arrays["timestamps"].astype(np.int64)
    symbols = tuple(protocol["dataset"]["fit_symbols"])
    feature_names = tuple(protocol["dataset"]["feature_names"])
    configuration = next(
        row
        for row in protocol["configurations"]
        if row["id"] == binding.configuration_id
    )
    horizon_hours = configuration["horizon_hours"]
    horizon_seconds = horizon_hours * 3600
    feature_indices = list(range(len(feature_names)))
    variance = 0.01
    vintages = []
    for ordinal, start_index in enumerate(protocol["actor"]["vintage_starts"]):
        start_ns = int(timestamps[start_index])
        cutoff_ns = start_ns - 1
        digest = hashlib.sha256(f"synthetic-vintage-{ordinal}".encode()).hexdigest()
        vintages.append(
            {
                "schema": "simple_return_ridge_vintage_v1",
                "return_unit": "expected_simple_return",
                "forecast_kind": "regularized_linear_projection",
                "valuation_basis": "same_close_price_return",
                "calibration_state": "uncalibrated",
                "variance_kind": "fit_prefix_marginal_variance",
                "block": {
                    "fit_cutoff": cutoff_ns,
                    "model_fit_time": cutoff_ns,
                    "prediction_start": start_ns,
                    "prediction_stop": int(timestamps[start_index + 1]),
                    "inference_delay_seconds": 0,
                },
                "training": {
                    "schema": "simple_return_training_scope_v1",
                    "log_training": {
                        "schema": "forecast_training_scope_v1",
                        "feature_indices": feature_indices,
                        "feature_names": list(feature_names),
                        "features": [[0.0] * len(feature_names)],
                        "labels": [0.0],
                        "sample_weights": [1.0],
                        "horizon_hours": horizon_hours,
                        "fit_cutoff": cutoff_ns,
                        "trace": {},
                    },
                    "labels": [0.0],
                    "return_unit": "expected_simple_return",
                    "valuation_basis": "same_close_price_return",
                    "variance_kind": "fit_prefix_marginal_variance",
                    "fit_prefix_marginal_variance": variance,
                },
                "model": {
                    "feature_indices": feature_indices,
                    "feature_mean": [0.0] * len(feature_names),
                    "feature_scale": [1.0] * len(feature_names),
                    "coefficients": [0.0] * len(feature_names),
                    "intercept": 0.0,
                    "horizon_hours": horizon_hours,
                    "alpha": configuration["ridge_alpha"],
                    "n_samples": 1,
                    "fit_cutoff": cutoff_ns,
                    "return_unit": "expected_simple_return",
                    "valuation_basis": "same_close_price_return",
                    "variance_kind": "fit_prefix_marginal_variance",
                    "fit_prefix_marginal_variance": variance,
                },
                "prediction_symbols": list(symbols),
                "digest": digest,
            }
        )
    decision_index = binding.window.actor_start
    decision_ns = int(timestamps[decision_index])
    selected = binding.selected_symbol_index
    first_vintage = vintages[0]
    packet = {
        "schema": "simple_return_packet_v1",
        "symbol": symbols[selected],
        "as_of": decision_ns,
        "source_available_at": int(
            arrays["available_at"][decision_index, selected].astype(np.int64)
        ),
        "feature_values": [
            float(value) for value in arrays["features"][decision_index, selected]
        ],
        "decision_close": float(arrays["close"][decision_index, selected]),
        "snapshot_digest": hashlib.sha256(b"synthetic-snapshot").hexdigest(),
        "forecast_available_at": decision_ns,
        "horizon_end": decision_ns + horizon_seconds * 1_000_000_000,
        "horizon_seconds": horizon_seconds,
        "expected_simple_return": 0.0,
        "fit_prefix_marginal_variance": variance,
        "vintage_digest": first_vintage["digest"],
        "return_unit": "expected_simple_return",
        "valuation_basis": "same_close_price_return",
        "variance_kind": "fit_prefix_marginal_variance",
        "forecast_kind": "regularized_linear_projection",
        "digest": hashlib.sha256(b"synthetic-first-packet").hexdigest(),
    }
    stream = {
        "schema": "simple_return_ridge_stream_v1",
        "availability_semantics": "declared_simulation_v1",
        "dataset_id": content.dataset_id,
        "vintages": vintages,
        "packets": [packet],
        "causal_scope_digest": hashlib.sha256(b"synthetic-scope").hexdigest(),
    }
    return json.dumps(
        stream, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _native_actor_row(advance: ReplayAdvance, replay) -> dict[str, object]:
    assert advance.actor_context_bytes is not None
    assert advance.actor_proposal_bytes is not None
    assert advance.actor_decision_bytes is not None
    assert advance.actor_action_bytes is not None
    context = json.loads(advance.actor_context_bytes)
    proposal = json.loads(advance.actor_proposal_bytes)
    decision = json.loads(advance.actor_decision_bytes)
    action = json.loads(advance.actor_action_bytes)

    def native_clock(value: str) -> int:
        return int(np.datetime64(value, "ns").astype(np.int64))

    context["decision_time"] = native_clock(context["decision_time"])
    inputs = dict(proposal["inputs"])
    for name in ("decision_time", "available_at", "horizon_end"):
        inputs[name] = native_clock(inputs[name])
    baseline = {
        "decision_digest": advance.actor_proposal_digest,
        "context": context,
        "inputs": inputs,
        "allocator": proposal["allocator"],
        "target_weight": proposal["target"],
        "objective_value": proposal["objective_value"],
    }
    native_decision = {
        "baseline": baseline,
        "action_contract": {
            name: value
            for name, value in decision["action_contract"].items()
            if name != "schema"
        },
        "feature_names": decision["feature_names"],
        "feature_values": decision["feature_values"],
        "max_drawdown": decision["max_drawdown"],
        "initial_capital": decision["initial_capital"],
        "remaining_steps": decision["remaining_steps"],
        "pending_gross": decision["pending_gross"],
        "pending_count": decision["pending_count"],
    }
    native_proposal = {
        "decision": native_decision,
        "raw_action": action["raw_action"],
        "action": action["action"],
        "raw_target_weight": action["raw_target_weight"],
        "target_weight": action["target_weight"],
        "is_hold": action["is_hold"],
    }
    snapshot = replay.account_snapshot
    processing = advance.processing_index
    decision_time = int(
        replay.content.arrays["timestamps"][advance.decision_index].astype(np.int64)
    )
    processing_time = int(
        replay.content.arrays["timestamps"][processing].astype(np.int64)
    )
    book = {
        "as_of_index": processing,
        "as_of_dataset_id": replay.content.dataset_id,
        "exact_quantities": [str(value) for value in snapshot.exact_quantities],
        "cash": snapshot.cash,
        "equity": snapshot.wealth,
        "peak_value": snapshot.peak_value,
        "max_drawdown": snapshot.max_drawdown,
        "total_cost": snapshot.total_cost,
        "funding_pnl": snapshot.funding_pnl,
        "borrow_cost": snapshot.borrow_cost,
        "turnover_total": snapshot.turnover_total,
        "margin_used": snapshot.margin_used,
        "maintenance_margin": snapshot.maintenance_margin,
        "maintenance_requirement": snapshot.maintenance_requirement,
        "margin_deficit": snapshot.margin_deficit,
        "fill_count": snapshot.fill_count,
        "rebalance_events": snapshot.rebalance_events,
        "insolvent": snapshot.insolvent,
        "termination_reason": snapshot.termination_reason,
        "quantities": list(snapshot.projected_quantities),
        "mark_prices": list(snapshot.marks),
        "contract_multipliers": list(snapshot.multipliers),
    }
    native = {
        "decision_index": advance.decision_index,
        "decision_time_ns": decision_time,
        "processing_index": processing,
        "processing_time_ns": processing_time,
        "symbol_index": replay.selected_symbol_index,
        "execution": {
            "next_index": processing,
            "bars_advanced": 1,
            "interval_net_return": snapshot.returns_history[-1],
        },
        "proposal": native_proposal,
        "order_events": [],
        "active_orders": [],
        "terminal_order_reasons": [],
        "book": book,
    }
    return {
        "decision_index": advance.decision_index,
        "decision_time_ns": decision_time,
        "native_validation_failure": None,
        "terminated": False,
        "truncated": False,
        "native": native,
    }


def _indexed_native_output(tmp_path: Path, receipt: dict[str, object], forecast: bytes):
    from tools.market_b4_oracle.output_index import (
        ArtifactPin,
        load_pinned_output_index,
    )

    receipt_bytes = gzip.compress(
        json.dumps(
            receipt, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode(),
        mtime=0,
    )
    files = {
        "forecasts/arm.json": forecast,
        "receipts/arm.json.gz": receipt_bytes,
    }
    root = tmp_path / "indexed-native-ledger"
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest = json.dumps(
        {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in files.items()
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (root / "output-manifest.json").write_bytes(manifest)
    return load_pinned_output_index(
        root, ArtifactPin(len(manifest), hashlib.sha256(manifest).hexdigest())
    )


def test_native_consumer_applies_reference_after_strict_events_and_q_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.tools.market_b4_oracle.test_offline_replay_consumer import (
        _native_event_consumer_fixture,
    )
    from tests.tools.market_replay_oracle.test_event_chain import (
        _actor_row,
        _actor_stream,
        _reserve_stream,
    )
    from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay

    module, exact_fill, binding, policy, pins = _native_event_consumer_fixture(
        tmp_path, monkeypatch
    )
    content = _complete_native_content(exact_fill, binding)
    pair = binding.protocol_pair
    assert pair is not None
    monkeypatch.setattr(
        module,
        "_readback_static_inputs",
        lambda **_kwargs: (pair, exact_fill, content),
    )
    forecast_bytes = _native_forecast_bytes(content, binding)
    forecast_index = FrozenForecastIndex(
        forecast_bytes,
        pinned_bytes=len(forecast_bytes),
        pinned_sha256=hashlib.sha256(forecast_bytes).hexdigest(),
    )
    replay = FrozenMarketReplay(
        content=content,
        policy=policy,
        binding=binding,
        reserve_plan_digest="5" * 64,
    )
    expected_advance = replay.advance(forecast_index=forecast_index, role="NONRL")
    assert not expected_advance.owned_fills
    assert not expected_advance.new_intent_ids
    first_row = _native_actor_row(expected_advance, replay)

    actor_rows = list(_actor_stream(exact_fill))
    actor_rows[0] = _actor_row(exact_fill, 1, events=[])
    actor_rows[0].update(
        {
            "terminated": False,
            "truncated": False,
            "native_validation_failure": None,
            "native": first_row["native"],
        }
    )
    output_index = _indexed_native_output(
        tmp_path,
        {
            "native_receipt": {"rows": actor_rows},
            "closing_rows": list(_reserve_stream(exact_fill)),
        },
        forecast_bytes,
    )

    phase_order: list[str] = []
    real_validator = module._validate_native_event_rows
    real_advance = FrozenMarketReplay.advance
    real_identity = module._compare_actor_identity
    real_effects = module._compare_event_effects
    real_orders = module._compare_orders
    real_book = module._compare_book
    real_ledger = module._NativeReferenceLedger
    completed_ledger: list[object] = []

    def validate_events(**kwargs: object) -> None:
        real_validator(**kwargs)
        phase_order.append("strict-v4-event-chain")

    def advance_one(self, **kwargs: object):
        advance = real_advance(self, **kwargs)
        phase_order.append("owned-q-advance")
        return advance

    def compare_identity(*args: object, **kwargs: object) -> None:
        real_identity(*args, **kwargs)
        phase_order.append("actor-identity")

    def compare_effects(*args: object, **kwargs: object) -> None:
        real_effects(*args, **kwargs)
        phase_order.append("native-fill-effects")

    def compare_orders(*args: object, **kwargs: object) -> None:
        real_orders(*args, **kwargs)
        phase_order.append("order-state")

    def compare_book(*args: object, **kwargs: object) -> None:
        real_book(*args, **kwargs)
        phase_order.append("account-book")

    class StopAfterReferenceRow(Exception):
        pass

    class OneRowReferenceLedger:
        def __init__(self, **kwargs: object) -> None:
            self.inner = real_ledger(**kwargs)

        def apply_advance(self, **kwargs: object) -> None:
            self.inner.apply_advance(**kwargs)
            completed_ledger.append(self.inner)
            phase_order.append("independent-reference")
            raise StopAfterReferenceRow

    monkeypatch.setattr(module, "_validate_native_event_rows", validate_events)
    monkeypatch.setattr(FrozenMarketReplay, "advance", advance_one)
    monkeypatch.setattr(module, "_compare_actor_identity", compare_identity)
    monkeypatch.setattr(module, "_compare_event_effects", compare_effects)
    monkeypatch.setattr(module, "_compare_orders", compare_orders)
    monkeypatch.setattr(module, "_compare_book", compare_book)
    monkeypatch.setattr(module, "_NativeReferenceLedger", OneRowReferenceLedger)

    with pytest.raises(StopAfterReferenceRow):
        module.consume_indexed_market_arm(
            output_index=output_index,
            receipt_name="receipts/arm.json.gz",
            forecast_name="forecasts/arm.json",
            content=content,
            policy=policy,
            binding=binding,
            role="NONRL",
            reserve_plan_digest="5" * 64,
            static_inputs=pins,
        )

    assert phase_order == [
        "strict-v4-event-chain",
        "owned-q-advance",
        "actor-identity",
        "native-fill-effects",
        "order-state",
        "account-book",
        "independent-reference",
    ]
    assert len(completed_ledger) == 1
    assert (
        completed_ledger[0].account.snapshot().exact_quantities
        == replay.account_snapshot.exact_quantities
    )
    assert_observed_account(
        completed_ledger[0].account.snapshot(),
        _observed_account(replay.account_snapshot),
    )
