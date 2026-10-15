"""Ordinary composition of public costs, native carriers and terminal plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from tier_d_protocol import CAPITAL, json_sha256

from trade_rl.artifacts import content_digest
from trade_rl.data.build.economics import ExecutionEconomicsProfile
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation_costs import (
    DeclaredAllocationCostRecipe,
    estimate_declared_horizon_costs,
)
from trade_rl.evaluation.allocation_global_execution import (
    declare_global_allocation_execution,
)
from trade_rl.evaluation.allocation_market_inputs import PreparedAllocationMarketInputs
from trade_rl.evaluation.allocation_settled_forecast_comparison import (
    TerminalAllocationNonRLControls,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    declare_terminal_allocation_execution,
)
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def execution_cost(scenario: str) -> ExecutionCostConfig:
    return ExecutionCostConfig(
        fee_rate=0.0,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0001,
        multiplier=2.0 if scenario == "COST2L0" else 1.0,
        max_participation_rate=0.05,
        processing_bar_volume_capacity=False,
        borrow_rate_multiplier=1.0,
        order_latency_bars=2 if scenario == "BASEL2" else 0,
        random_seed=0,
    )


def forecast_blocks(
    dataset: MarketDataset, starts: tuple[int, ...], stop: int
) -> tuple[ForecastBlock, ...]:
    return tuple(
        ForecastBlock(
            dataset.timestamps[start] - np.timedelta64(1, "ns"),
            dataset.timestamps[start],
            dataset.timestamps[start],
            dataset.timestamps[end],
        )
        for start, end in zip(starts, (*starts[1:], stop), strict=True)
    )


def execution_folds(starts: tuple[int, ...], stop: int) -> tuple[WalkForwardFold, ...]:
    # These type-required prefix slots have no checkpoint/HPO consumer here.
    return tuple(
        WalkForwardFold(
            i,
            IndexRange(0, start - 2),
            IndexRange(start - 2, start - 1),
            IndexRange(start - 1, start),
            IndexRange(start, end),
            0,
        )
        for i, (start, end) in enumerate(zip(starts, (*starts[1:], stop), strict=True))
    )


def cost_rows(
    dataset: MarketDataset, protocol: dict[str, Any], horizon: int, scenario: str
) -> tuple[HorizonCostEstimates, ...]:
    cost = execution_cost(scenario)
    recipe = DeclaredAllocationCostRecipe(
        execution_cost=cost,
        expected_execution_policy_digest=cost.execution_policy_digest,
        reference_notional=CAPITAL,
        assumptions_identity=json_sha256(protocol),
        assumptions_available_at=dataset.timestamps[0],
        fee_assumption="Declared Task18 generic fee .0005; overlay0; stress multiplier2.",
        spread_assumption="Declared Task18 full spread .0002; overlay0; stress multiplier2.",
        impact_assumption="Declared square-root participation coefficient .0001, not calibrated.",
        slippage_assumption="Deterministic zero stochastic slippage declaration.",
        borrow_assumption="Borrow available; zero declared borrow rate; multiplier1.",
        funding_source_identity=f"Declared bar-close-rate proxy:{dataset.dataset_id}",
        funding_coverage_start=dataset.timestamps[0],
        funding_coverage_stop=dataset.timestamps[-1],
        funding_lookback_hours=24,
        funding_minimum_event_count=2,
        funding_maximum_counted_bar_close_gap_hours=9.0,
        funding_maximum_counted_bar_close_age_hours=8.0,
        cash_zero_rationale="Nominal USDT quote cash assumed zero yield, USDT FX parity1.",
    )
    profile = ExecutionEconomicsProfile.from_payload(
        json.loads(dataset.identity_payload_json or "{}")["execution_economics"]
    )
    return estimate_declared_horizon_costs(
        dataset,
        decision_indices=tuple(
            range(protocol["actor"]["start"], protocol["actor"]["stop"])
        ),
        horizon_hours=horizon,
        recipe=recipe,
        declared_execution_economics=profile,
        expected_dataset_id=dataset.dataset_id,
    )


@dataclass(frozen=True)
class PairDeclaration:
    account: str
    scenario: str
    configuration: str
    symbol_index: int
    controls: TerminalAllocationNonRLControls


def carrier(
    prepared: PreparedAllocationMarketInputs,
    estimates: tuple[HorizonCostEstimates, ...],
    protocol: dict[str, Any],
    *,
    horizon: int,
    symbol_index: int,
    account: str,
    scenario: str,
    role: str,
) -> AllocationTradingEnv:
    dataset = prepared.dataset
    start, stop = protocol["actor"]["start"], protocol["actor"]["stop"]
    risk = PreTradeRiskConfig(**protocol["policy"]["risk"])
    allocator = AfterCostTargetAllocator(**protocol["policy"]["allocator"])
    cost = execution_cost(scenario)
    action = AllocationActionContract(
        mode="residual" if role == "nonrl" else "direct", scale=1.0
    )
    economics = MarketExecutor(
        dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    profile = AllocationRuntimeProfile(
        economics, content_digest(risk), CAPITAL, "USDT", 3600, (stop - start) * 3600
    )
    schema = AllocationObservationSchema(
        dataset.feature_names, 8, CAPITAL, stop - start
    )
    recipe = allocation_recipe_payload_v2(
        action,
        dataset.feature_names,
        allocator=allocator,
        expected_horizon_seconds=horizon * 3600,
        runtime_profile=profile,
        observation_schema=schema,
    )
    clock = FinancialClockContract(
        3600, 3600, 3600, (stop - start) * 3600, 32, 1.0, 0.95, "equity_delta_v1"
    )
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USDT", (CAPITAL,)),
        dataset.timestamps[start]
        .astype("datetime64[us]")
        .astype(datetime)
        .replace(tzinfo=UTC),
        dataset.timestamps[stop]
        .astype("datetime64[us]")
        .astype(datetime)
        .replace(tzinfo=UTC),
        "marked_continuation",
        economics,
        content_digest(risk),
        content_digest(recipe),
        0.20,
    )
    return AllocationTradingEnv(
        dataset,
        stream=prepared.stream,
        estimates=estimates,
        bound=BoundObjectiveClock(objective, clock),
        action_contract=action,
        allocator=allocator,
        execution_cost=cost,
        risk_config=risk,
        feature_indices=tuple(range(dataset.n_features)),
        symbol_index=symbol_index,
        start_index=start,
        stop_index=stop,
        account_id=account,
        expected_horizon_seconds=horizon * 3600,
        observation_schema=schema,
    )


def declare_pair(
    prepared: PreparedAllocationMarketInputs,
    estimates: tuple[HorizonCostEstimates, ...],
    protocol: dict[str, Any],
    *,
    horizon: int,
    symbol_index: int,
    account: str,
    scenario: str,
    configuration: str,
    provenance: dict[str, Any],
) -> PairDeclaration:
    folds = execution_folds(
        tuple(protocol["actor"]["vintage_starts"]), protocol["actor"]["stop"]
    )
    plans = []
    for role in ("nonrl", "cash"):
        env = carrier(
            prepared,
            estimates,
            protocol,
            horizon=horizon,
            symbol_index=symbol_index,
            account=account,
            scenario=scenario,
            role=role,
        )
        native = declare_global_allocation_execution(
            folds,
            env,
            kind=role,
            scenario=scenario,
            expected_implementation_digest=provenance["implementation_digest"],
            expected_runtime_digest=provenance["runtime_environment_digest"],
        )
        plans.append(
            declare_terminal_allocation_execution(
                folds,
                env,
                native,
                settlement_stop_index=protocol["terminal"]["stop_inclusive"],
            )
        )
    return PairDeclaration(
        account,
        scenario,
        configuration,
        symbol_index,
        TerminalAllocationNonRLControls(*plans),
    )


def synthetic_protocol(
    protocol: dict[str, Any], dataset: MarketDataset
) -> dict[str, Any]:
    result = json.loads(json.dumps(protocol))
    result["scope"] = "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY"
    result["accounts"] = ["ARTIFICIAL_H72_A10_A"]
    result["scenarios"] = ["BASEL2"]
    result["configurations"] = [
        {"id": "H72_RIDGE_A10", "horizon_hours": 72, "ridge_alpha": 10.0}
    ]
    result["dataset"].update(
        dataset_id=dataset.dataset_id,
        dimensions=[256, 2, 6],
        feature_names=list(dataset.feature_names),
        fit_symbols=list(dataset.symbols),
        development_start=str(dataset.timestamps[0]),
        development_stop_exclusive=str(dataset.timestamps[-1] + np.timedelta64(1, "h")),
    )
    result["actor"] = {"start": 128, "stop": 160, "rows": 32, "vintage_starts": [128]}
    result["terminal"] = {"stop_inclusive": 168, "advances": 8}
    result["budget"] = {
        "native_arms": 2,
        "dataset_loads": 2,
        "cost_producers": 1,
        "public_prepare_fits": 1,
        "ridge_vintages": 1,
        "total_native_advances": 80,
    }
    return result
