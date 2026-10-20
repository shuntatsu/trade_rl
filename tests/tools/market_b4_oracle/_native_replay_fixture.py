from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from types import MappingProxyType

import numpy as np

from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
from tools.market_b4_oracle.frozen_market_replay import (
    FrozenMarketBinding,
    FrozenMarketPolicy,
    FrozenMarketReplay,
    FrozenMarketWindow,
)
from tools.market_replay_oracle.artifact import VerifiedDatasetContent


def _public_native_fixture(
    tmp_path,
    selected_symbol: int,
    *,
    scenario: str = "COST2L0",
    role: str = "nonrl",
    horizon: int = 72,
    quote_volume: float = 10_000_000.0,
    task18_execution_rules: bool = False,
    rounding_partial_case: bool = False,
):
    """Build one frozen-profile artificial 32+8 arm and owned input pins."""
    from dataclasses import replace as dataclass_replace

    from tools.market_replay_oracle.identity import canonical
    from tools.tier_d_nonrl.tier_d_fixture import write_artificial_dataset
    from tools.tier_d_nonrl.tier_d_native import (
        carrier,
        cost_rows,
        execution_folds,
        forecast_blocks,
        synthetic_protocol,
    )
    from tools.tier_d_nonrl.tier_d_protocol import market_protocol
    from trade_rl.data import write_market_dataset_files
    from trade_rl.data.build.builder import MarketDatasetBuilder
    from trade_rl.data.build.economics import ExecutionEconomicsProfile
    from trade_rl.data.contracts import (
        FeatureKind,
        FeatureSpec,
        InstrumentContract,
        MarketBuildConfig,
        VolumeUnit,
    )
    from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
    from trade_rl.evaluation.allocation_global_execution import (
        declare_global_allocation_execution,
    )
    from trade_rl.evaluation.allocation_market_inputs import (
        prepare_allocation_market_inputs,
    )
    from trade_rl.evaluation.allocation_terminal_execution import (
        declare_terminal_allocation_execution,
        run_terminal_allocation_execution,
    )
    from trade_rl.evaluation.runs import build_candidate_run_provenance

    source_root = tmp_path / f"source-{selected_symbol}-{scenario}-{role}-h{horizon}"
    original = write_artificial_dataset(source_root)
    if quote_volume != 10_000_000.0 or task18_execution_rules or rounding_partial_case:
        source_identity = json.loads(original.identity_payload_json or "{}")
        source_economics = ExecutionEconomicsProfile.from_payload(
            source_identity["execution_economics"]
        )
        raw_values = {}
        for symbol_index, symbol in enumerate(original.symbols):
            open_prices = np.array(original.open[:, symbol_index], copy=True)
            high_prices = np.array(original.high[:, symbol_index], copy=True)
            low_prices = np.array(original.low[:, symbol_index], copy=True)
            close_prices = np.array(original.close[:, symbol_index], copy=True)
            if symbol_index == selected_symbol and rounding_partial_case:
                tick_size = 0.1 if symbol_index == 0 else 0.01
                prior_mark = float(close_prices[128])
                off_grid_mark = (
                    math.floor(prior_mark / tick_size) + 1
                ) * tick_size - tick_size * 1e-6
                close_prices[128] = off_grid_mark
                open_prices[129] = off_grid_mark
                for index in (128, 129):
                    high_prices[index] = max(high_prices[index], off_grid_mark)
                    low_prices[index] = min(low_prices[index], off_grid_mark)
            event_counts = original.resolved_array("funding_event_count")[
                :, symbol_index
            ]
            raw_values[symbol] = RawMarketSeries(
                timestamps=original.timestamps,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices,
                volume=np.full(256, quote_volume, dtype=np.float64),
                funding_rate=original.funding_rate[:, symbol_index],
                tradable=original.tradable[:, symbol_index],
                funding_available=event_counts > 0,
                funding_event_count=event_counts,
                funding_price_rate=original.resolved_array("funding_price_rate")[
                    :, symbol_index
                ],
            )
        feature_specs = (
            FeatureSpec("artificial_return_1", FeatureKind.LOG_RETURN),
            FeatureSpec("artificial_return_24", FeatureKind.LOG_RETURN, lookback=24),
            FeatureSpec("artificial_body", FeatureKind.BODY_RETURN),
            FeatureSpec("artificial_range", FeatureKind.HIGH_LOW_RANGE),
            FeatureSpec(
                "artificial_volatility", FeatureKind.REALIZED_VOLATILITY, lookback=24
            ),
            FeatureSpec(
                "artificial_volume_z",
                FeatureKind.VOLUME_ZSCORE,
                lookback=24,
                min_periods=24,
            ),
        )
        original = MarketDatasetBuilder(
            MarketBuildConfig(base_timeframe="1h", features=feature_specs)
        ).build(
            InMemoryMarketDataSource(raw_values),
            tuple(
                InstrumentContract(
                    symbol,
                    datetime(2024, 1, 1, tzinfo=UTC),
                    volume_unit=VolumeUnit.QUOTE_NOTIONAL,
                    tick_size=(0.1 if symbol_index == 0 else 0.01)
                    if task18_execution_rules
                    else 0.0,
                    lot_size=0.001 if task18_execution_rules else 0.0,
                    minimum_notional=(50.0 if symbol_index == 0 else 20.0)
                    if task18_execution_rules
                    else 0.0,
                )
                for symbol_index, symbol in enumerate(original.symbols)
            ),
            execution_economics=source_economics,
        )
    economics = ExecutionEconomicsProfile(
        name="artificial Task18 declared execution economics",
        fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )
    identity = json.loads(original.identity_payload_json or "{}")
    identity["execution_economics"] = economics.to_payload()
    identity["scope"] = "ARTIFICIAL_COMPARISON_ONLY"
    zero_borrow = dataclass_replace(
        original,
        borrow_rate=np.zeros_like(original.borrow_rate),
        volume=np.full_like(original.volume, quote_volume, dtype=np.float64),
        identity_payload_json=None,
    ).with_content_identity(identity)
    dataset_root = tmp_path / f"dataset-{selected_symbol}-{scenario}-{role}-h{horizon}"
    write_market_dataset_files(dataset_root, zero_borrow)
    base_protocol = market_protocol()
    protocol = synthetic_protocol(base_protocol, zero_borrow)
    protocol["scope"] = "ARTIFICIAL_COMPARISON_ONLY"
    configuration_id = f"TEST_H{horizon}_RIDGE_A1"
    protocol["configurations"] = [
        {"id": configuration_id, "horizon_hours": horizon, "ridge_alpha": 1.0}
    ]
    accounts = [f"{configuration_id}_{symbol}" for symbol in zero_borrow.symbols]
    protocol["accounts"] = accounts
    protocol["scenarios"] = [scenario]
    protocol["roles"] = [role]
    protocol["actor"] = {
        "start": 128,
        "stop": 160,
        "rows": 32,
        "vintage_starts": [128],
    }
    protocol["terminal"] = {"stop_inclusive": 168, "advances": 8}
    protocol["dataset"]["dimensions"] = [256, 2, 6]
    protocol["dataset"]["development_start"] = str(zero_borrow.timestamps[0])
    protocol["dataset"]["development_stop_exclusive"] = str(
        zero_borrow.timestamps[-1] + np.timedelta64(1, "h")
    )
    protocol["economics"].update(
        borrow_rate=0.0,
        declaration="Artificial Task18 zero-borrow profile",
    )

    native_assumptions_identity = hashlib.sha256(canonical(protocol)).hexdigest()
    # The native runner reuses BASEL0 forecast-cost rows for the BASEL2
    # execution-latency stress; only COST2L0 has a distinct cost-row recipe.
    cost_scenario = "COST2L0" if scenario == "COST2L0" else "BASEL0"
    estimates = cost_rows(zero_borrow, protocol, horizon, cost_scenario)
    stream_path = (
        tmp_path / f"stream-{selected_symbol}-{scenario}-{role}-h{horizon}.json"
    )
    prepared = prepare_allocation_market_inputs(
        dataset_root,
        stream_path,
        expected_dataset_id=zero_borrow.dataset_id,
        development_start=zero_borrow.timestamps[0],
        development_stop=zero_borrow.timestamps[-1] + np.timedelta64(1, "h"),
        feature_names=zero_borrow.feature_names,
        fit_symbols=zero_borrow.symbols,
        blocks=forecast_blocks(zero_borrow, (128,), 160),
        horizon_hours=horizon,
        alpha=1.0,
        cost_rows=tuple(row.payload() for row in estimates),
    )
    account_id = accounts[selected_symbol]
    env = carrier(
        prepared,
        tuple(
            row
            for row in estimates
            if row.symbol == zero_borrow.symbols[selected_symbol]
        ),
        protocol,
        horizon=horizon,
        symbol_index=selected_symbol,
        account=account_id,
        scenario=scenario,
        role=role,
    )
    folds = execution_folds((128,), 160)
    provenance = build_candidate_run_provenance()
    native_plan = declare_global_allocation_execution(
        folds,
        env,
        kind=role,
        scenario=scenario,
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_digest=provenance["runtime_environment_digest"],
    )
    terminal_plan = declare_terminal_allocation_execution(
        folds, env, native_plan, settlement_stop_index=168
    )
    native = run_terminal_allocation_execution(folds, env, terminal_plan)

    arrays = dict(zero_borrow.identity_arrays())
    arrays["asset_active"] = zero_borrow.resolved_array("asset_active")
    verified = VerifiedDatasetContent(
        zero_borrow.dataset_id,
        MappingProxyType(arrays),
        MappingProxyType(
            {
                "symbols": zero_borrow.symbols,
                "feature_names": zero_borrow.feature_names,
                "global_feature_names": zero_borrow.global_feature_names,
                "volume_units": tuple(unit.value for unit in zero_borrow.volume_units),
                "calendar_kind": zero_borrow.calendar_kind.value,
                "nominal_bar_hours": zero_borrow.nominal_bar_hours,
                "periods_per_year": zero_borrow.periods_per_year,
                "identity_payload_json": zero_borrow.identity_payload_json,
            }
        ),
    )
    binding = FrozenMarketBinding(
        protocol_canonical=canonical(protocol),
        account_id=account_id,
        configuration_id=configuration_id,
        scenario=scenario,
        native_role=role,
        selected_symbol_index=selected_symbol,
        expected_horizon_seconds=horizon * 3600,
        observation_feature_indices=tuple(range(zero_borrow.n_features)),
        window=FrozenMarketWindow(128, 160, 168),
        comparison_only=True,
    )
    policy = FrozenMarketPolicy.from_effective_profile(
        dataset_id=zero_borrow.dataset_id,
        profile=binding.effective_profile,
        risk=protocol["policy"]["risk"],
    )
    raw_stream = json.dumps(
        prepared.stream.payload(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    forecast_index = FrozenForecastIndex(
        raw_stream,
        pinned_bytes=len(raw_stream),
        pinned_sha256=hashlib.sha256(raw_stream).hexdigest(),
    )
    owned = FrozenMarketReplay(
        content=verified,
        policy=policy,
        binding=binding,
        reserve_plan_digest=terminal_plan.digest,
    )
    native_estimate = next(
        row.payload()
        for row in estimates
        if row.symbol == zero_borrow.symbols[selected_symbol]
        and row.decision_time == zero_borrow.timestamps[128]
    )
    return (
        owned,
        forecast_index,
        native.payload,
        native_estimate,
        native_assumptions_identity,
    )


def _native_actor_identity_hashes(native_actor):
    """Rebuild the native baseline, decision, action and selected actor hashes."""
    from trade_rl.artifacts import content_digest

    proposal = native_actor["proposal"]
    decision = proposal["decision"]
    baseline = decision["baseline"]
    allocator = baseline["allocator"]
    current_weight = baseline["context"]["current_weight"]
    lower = allocator["lower_weight"]
    upper = allocator["upper_weight"]
    if allocator["max_turnover"] is not None:
        lower = max(lower, current_weight - allocator["max_turnover"])
        upper = min(upper, current_weight + allocator["max_turnover"])
    decision_digest = content_digest(
        {
            "schema": "allocation_decision_v1",
            "baseline": baseline["decision_digest"],
            "action_contract": {
                "schema": "allocation_action_v1",
                **decision["action_contract"],
            },
            "feasible_bounds": (float(lower), float(upper)),
            "feature_names": decision["feature_names"],
            "feature_values": decision["feature_values"],
            "max_drawdown": decision["max_drawdown"],
            "initial_capital": decision["initial_capital"],
            "remaining_steps": decision["remaining_steps"],
            "pending_gross": decision["pending_gross"],
            "pending_count": decision["pending_count"],
        }
    )
    action_digest = content_digest(
        {
            "schema": "allocation_action_proposal_v1",
            "decision": decision_digest,
            "raw_action": proposal["raw_action"],
            "action": proposal["action"],
            "raw_target_weight": proposal["raw_target_weight"],
            "target_weight": proposal["target_weight"],
            "is_hold": proposal["is_hold"],
        }
    )
    selected_identity = (
        baseline["decision_digest"]
        if proposal["target_weight"] == baseline["target_weight"]
        else action_digest
    )
    return {
        "baseline": baseline["decision_digest"],
        "context_state": baseline["context"]["state_digest"],
        "decision": decision_digest,
        "action": action_digest,
        "selected": selected_identity,
    }
