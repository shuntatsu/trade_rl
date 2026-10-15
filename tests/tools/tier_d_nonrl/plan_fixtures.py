"""Manufactured declaration JSON; no Dataset, forecast fitting, or execution."""

from datetime import UTC, datetime, timedelta

from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_fold_plan_digest,
)
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionPlan,
)
from trade_rl.evaluation.allocation_nonrl_walk_forward import (
    allocation_nonrl_rule_digest,
)
from trade_rl.evaluation.allocation_scenario_identity import (
    allocation_candidate_recipe_digest,
)
from trade_rl.evaluation.allocation_settled_forecast_comparison import (
    SettledNonRLComparisonPlan,
    TerminalAllocationNonRLControls,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    TerminalAllocationExecutionPlan,
)
from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.evaluation.robustness.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.rl.allocation_fee_stress import retained_debt_economics_digest
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


def matrix_payload(common_overrides=None):
    capital = 100000.0
    pin = "a" * 64
    cost = ExecutionCostConfig().execution_policy_payload()
    economics = retained_debt_economics_digest(cost)
    epoch = datetime(2025, 1, 1, tzinfo=UTC)
    clocks = [int((epoch + timedelta(hours=h)).timestamp()) * 10**9 for h in range(11)]
    fold = WalkForwardFold(
        0, IndexRange(0, 1), IndexRange(1, 2), IndexRange(2, 3), IndexRange(6, 8), 0
    )
    fold_payload = {
        "fold_index": 0,
        "train": [0, 1],
        "checkpoint_validation": [1, 2],
        "configuration_selection": [2, 3],
        "test": [6, 8],
        "purge_bars": 0,
    }
    common = dict.fromkeys(
        (
            "source_digest",
            "dataset_id",
            "objective_digest",
            "clock_digest",
            "forecast_context_digest",
            "cost_beliefs_digest",
            "risk_digest",
            "implementation_digest",
            "runtime_environment_digest",
        ),
        pin,
    )
    common.update(
        account_id="A",
        symbol_index=0,
        range=[6, 8],
        source_clocks_ns=clocks[6:9],
        horizon_steps=2,
        initial_capital=capital,
        fold_plan_digest=allocation_fold_plan_digest((fold,)),
        fold_plan=[fold_payload],
        economics_digest=economics,
        scenario="BASEL0",
        source_semantics="declared_finite_envelope_not_getter_usage_or_origin",
    )
    common.update(common_overrides or {})
    plans = []
    for role in ("nonrl", "cash"):
        recipe = allocation_recipe_payload_v2(
            AllocationActionContract(mode="residual" if role == "nonrl" else "direct"),
            ("feature",),
            allocator=AfterCostTargetAllocator(),
            expected_horizon_seconds=86400,
            runtime_profile=AllocationRuntimeProfile(
                economics, pin, capital, "USDT", 3600, 7200
            ),
            observation_schema=AllocationObservationSchema(("feature",), 8, capital, 2),
        )
        candidate = allocation_candidate_recipe_digest(recipe)
        policy = (
            allocation_nonrl_rule_digest(candidate)
            if role == "nonrl"
            else content_digest(
                {
                    "schema": "allocation_cash_control_policy_v1",
                    "action_semantics": "quantity_hold_current_v1",
                    "raw_action": 0,
                    "carrier_recipe_digest": candidate,
                }
            )
        )
        native = GlobalAllocationExecutionPlan.from_payload(
            {
                "schema": "allocation_global_execution_plan_v1",
                "common": common,
                "cell": {
                    "kind": role,
                    "seed": None,
                    "reset_seed": None if role == "nonrl" else 0,
                    "original_recipe": recipe,
                    "original_recipe_digest": content_digest(recipe),
                    "original_candidate_digest": candidate,
                    "runtime": {
                        "recipe": recipe,
                        "cost_payload": cost,
                        "source_context_digest": pin,
                    },
                    "runtime_recipe_digest": content_digest(recipe),
                    "artifacts": [],
                    "control_policy_digest": policy,
                    "fee": None,
                },
            }
        )
        clock = FinancialClockContract(
            3600, 3600, 3600, 14400, 32, 1.0, 0.95, "equity_delta_v1"
        )
        terminal_recipe = {
            "schema": "allocation_terminal_recipe_v1",
            "native_plan_digest": content_digest(native.payload),
            "policy_range": [6, 8],
            "closing_range": [8, 10],
            "closing_clocks_ns": clocks[8:11],
            "closing_source_digest": pin,
            "clock": clock.payload(),
            "terminal_valuation": "settled",
            "target": "zero_through_unchanged_hard_risk",
            "coverage": "all_reserved_processing_bars_even_when_flat",
        }
        objective = ObjectiveContract(
            CapitalContract("independent_symbol", "USDT", (capital,)),
            epoch + timedelta(hours=6),
            epoch + timedelta(hours=10),
            "settled",
            economics,
            pin,
            content_digest(terminal_recipe),
        )
        plans.append(
            TerminalAllocationExecutionPlan(
                native,
                10,
                tuple(clocks[8:11]),
                pin,
                BoundObjectiveClock(objective, clock),
            )
        )
    return SettledNonRLComparisonPlan(
        (TerminalAllocationNonRLControls(*plans),), ("A",), ("BASEL0",)
    ).payload


def costs_payload():
    return {
        "schema": "tier_d_cost_tables_v1",
        "tables": [
            {
                "horizon_hours": 24,
                "scenario": "BASEL0",
                "rows": [
                    {
                        "schema": "horizon_cost_estimates_v1",
                        "valuation_basis": "same_close_price_return",
                        "cost_basis": "declared_initial_notional_rates",
                        "symbol": "ARTIFICIAL",
                        "decision_time": 1735711200000000000,
                        "available_at": 1735711200000000000,
                        "horizon_end": 1735797600000000000,
                        "source_identity": "a" * 64,
                        "buy_cost": 0.001,
                        "sell_cost": 0.001,
                        "exit_cost": 0.001,
                        "funding_return": 0.0001,
                        "borrow_return": 0.0,
                        "cash_return": 0.0,
                    }
                ],
            }
        ],
    }
