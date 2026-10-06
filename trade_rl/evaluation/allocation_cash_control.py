"""Explicit all-cash control on the canonical continuous allocation account."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_comparison import (
    AllocationComparisonContract,
    AllocationValidity,
)
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_comparison_oos_source_digest,
    allocation_continuous_execution_summary_digest,
    allocation_continuous_ledger_summary_digest,
    canonical_continuous_allocation_metrics,
    validate_continuous_allocation_comparison_context,
)
from trade_rl.evaluation.allocation_selection import AllocationCashReference
from trade_rl.evaluation.rl_allocation.continuous_walk_forward import (
    AllocationFoldPolicy,
    ContinuousAllocationWalkForwardResult,
    run_continuous_allocation_walk_forward,
)
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.robustness.walk_forward.folds import WalkForwardFold


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _closed_mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def _cash_policy_digest_for_recipe(recipe_digest: str) -> str:
    require_sha256(recipe_digest, field="carrier_recipe_digest")
    return content_digest(
        {
            "schema": "allocation_cash_control_policy_v1",
            "action_semantics": "quantity_hold_current_v1",
            "raw_action": 0,
            "carrier_recipe_digest": recipe_digest,
        }
    )


def allocation_cash_control_policy_digest(
    contract: AllocationComparisonContract,
) -> str:
    """Fixed seedless control identity, independent of scenario outcomes."""
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("cash control policy requires comparison contract")
    return _cash_policy_digest_for_recipe(contract.direct_recipe_digest)


@dataclass(frozen=True, slots=True)
class AllocationCashControlEvidence:
    contract_digest: str
    scenario: str
    control_policy_digest: str
    carrier_recipe_digest: str
    oos_source_digest: str
    ledger_digest: str
    execution_digest: str
    validity_evidence_digest: str
    opening_state_digest: str
    closing_state_digest: str
    terminal_profit_rate: float
    max_drawdown: float
    validity: AllocationValidity
    coverage_complete: bool
    termination_reason: str | None

    def __post_init__(self) -> None:
        for field in (
            "contract_digest",
            "control_policy_digest",
            "carrier_recipe_digest",
            "oos_source_digest",
            "ledger_digest",
            "execution_digest",
            "validity_evidence_digest",
            "opening_state_digest",
            "closing_state_digest",
        ):
            require_sha256(getattr(self, field), field=field)
        object.__setattr__(
            self,
            "scenario",
            require_non_empty(self.scenario, field="scenario"),
        )
        object.__setattr__(
            self,
            "terminal_profit_rate",
            _finite(self.terminal_profit_rate, "terminal_profit_rate"),
        )
        drawdown = _finite(self.max_drawdown, "max_drawdown")
        if drawdown < 0.0:
            raise ValueError("max_drawdown must be non-negative")
        object.__setattr__(self, "max_drawdown", drawdown)
        if type(self.validity) is not AllocationValidity:
            raise ValueError("validity must be AllocationValidity")
        if type(self.coverage_complete) is not bool:
            raise ValueError("coverage_complete must be a native bool")
        if self.termination_reason is not None:
            object.__setattr__(
                self,
                "termination_reason",
                require_non_empty(
                    self.termination_reason,
                    field="termination_reason",
                ),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_cash_control_evidence_v1",
            "contract_digest": self.contract_digest,
            "scenario": self.scenario,
            "control_policy_digest": self.control_policy_digest,
            "carrier_recipe_digest": self.carrier_recipe_digest,
            "oos_source_digest": self.oos_source_digest,
            "ledger_digest": self.ledger_digest,
            "execution_digest": self.execution_digest,
            "validity_evidence_digest": self.validity_evidence_digest,
            "opening_state_digest": self.opening_state_digest,
            "closing_state_digest": self.closing_state_digest,
            "terminal_profit_rate": self.terminal_profit_rate,
            "max_drawdown": self.max_drawdown,
            "validity": self.validity.value,
            "coverage_complete": self.coverage_complete,
            "termination_reason": self.termination_reason,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    @classmethod
    def from_payload(cls, value: object) -> AllocationCashControlEvidence:
        data = _closed_mapping(
            value,
            {
                "schema",
                "contract_digest",
                "scenario",
                "control_policy_digest",
                "carrier_recipe_digest",
                "oos_source_digest",
                "ledger_digest",
                "execution_digest",
                "validity_evidence_digest",
                "opening_state_digest",
                "closing_state_digest",
                "terminal_profit_rate",
                "max_drawdown",
                "validity",
                "coverage_complete",
                "termination_reason",
            },
            "allocation cash control evidence",
        )
        if data["schema"] != "allocation_cash_control_evidence_v1":
            raise ValueError("unsupported allocation cash control evidence schema")
        try:
            result = cls(
                contract_digest=data["contract_digest"],
                scenario=data["scenario"],
                control_policy_digest=data["control_policy_digest"],
                carrier_recipe_digest=data["carrier_recipe_digest"],
                oos_source_digest=data["oos_source_digest"],
                ledger_digest=data["ledger_digest"],
                execution_digest=data["execution_digest"],
                validity_evidence_digest=data["validity_evidence_digest"],
                opening_state_digest=data["opening_state_digest"],
                closing_state_digest=data["closing_state_digest"],
                terminal_profit_rate=data["terminal_profit_rate"],
                max_drawdown=data["max_drawdown"],
                validity=AllocationValidity(data["validity"]),
                coverage_complete=data["coverage_complete"],
                termination_reason=data["termination_reason"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("allocation cash control evidence is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("cash control evidence differs from canonical declaration")
        return result


def _validate_cash_result(
    environments: tuple[AllocationTradingEnv, ...],
    result: ContinuousAllocationWalkForwardResult,
    *,
    expected_policy_digest: str,
) -> None:
    if result.policy_digests != (expected_policy_digest,) * len(environments):
        raise ValueError("cash control result used another policy identity")
    for env, fold_result in zip(environments, result.folds, strict=True):
        if any(value != 0 for value in env.book.exact_quantities):
            raise ValueError("cash control must remain flat")
        if env.order_book.active_orders:
            raise ValueError("cash control must not retain active orders")
        diagnostics = fold_result.diagnostics
        if (
            diagnostics.n_trades != 0
            or abs(diagnostics.turnover_total) > 1e-12
            or abs(diagnostics.total_cost) > 1e-12
            or abs(diagnostics.funding_pnl) > 1e-12
            or abs(diagnostics.borrow_cost) > 1e-12
        ):
            raise ValueError("cash control must not trade or carry position economics")


def allocation_cash_reference(
    evidence: AllocationCashControlEvidence,
) -> AllocationCashReference:
    """Project verified cash-control evidence into the pure selection boundary."""
    if type(evidence) is not AllocationCashControlEvidence:
        raise ValueError("cash reference projection requires cash-control evidence")
    return AllocationCashReference(
        contract_digest=evidence.contract_digest,
        scenario=evidence.scenario,
        control_evidence_digest=evidence.digest,
        oos_source_digest=evidence.oos_source_digest,
        opening_state_digest=evidence.opening_state_digest,
        terminal_profit_rate=evidence.terminal_profit_rate,
        max_drawdown=evidence.max_drawdown,
        validity=evidence.validity,
        coverage_complete=evidence.coverage_complete,
        termination_reason=evidence.termination_reason,
    )


def run_continuous_allocation_cash_control(
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
) -> ContinuousAllocationWalkForwardResult:
    """Run fixed HOLD-current from a flat account through the continuous OOS chain."""
    if (
        type(environments) is not tuple
        or not environments
        or any(type(env) is not AllocationTradingEnv for env in environments)
    ):
        raise ValueError("cash control requires immutable allocation environments")
    recipe = environments[0].recipe_digest
    if any(env.recipe_digest != recipe for env in environments):
        raise ValueError("cash control carrier recipe must be identical across folds")
    policy_digest = _cash_policy_digest_for_recipe(recipe)
    policies = tuple(
        AllocationFoldPolicy(
            policy_digest=policy_digest,
            recipe_digest=env.recipe_digest,
            action=lambda _observation, _recipe: 0,
        )
        for env in environments
    )
    result = run_continuous_allocation_walk_forward(
        folds,
        environments,
        policies,
        reset_seed=0,
    )
    _validate_cash_result(
        environments,
        result,
        expected_policy_digest=policy_digest,
    )
    return result


def build_allocation_cash_control_evidence(
    contract: AllocationComparisonContract,
    *,
    scenario: str,
    folds: tuple[WalkForwardFold, ...],
    environments: tuple[AllocationTradingEnv, ...],
    result: ContinuousAllocationWalkForwardResult,
    validity_evidence_digest: str,
    validity: AllocationValidity = AllocationValidity.VALID,
) -> AllocationCashControlEvidence:
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("cash control evidence requires comparison contract")
    require_sha256(validity_evidence_digest, field="validity_evidence_digest")
    if type(validity) is not AllocationValidity:
        raise ValueError("cash control validity must be AllocationValidity")
    validate_continuous_allocation_comparison_context(
        contract,
        expected_recipe_digest=contract.direct_recipe_digest,
        scenario=scenario,
        folds=folds,
        environments=environments,
        result=result,
    )
    policy_digest = allocation_cash_control_policy_digest(contract)
    _validate_cash_result(
        environments,
        result,
        expected_policy_digest=policy_digest,
    )
    terminal_profit, max_drawdown = canonical_continuous_allocation_metrics(
        contract,
        environments,
        result,
    )
    source_digest = allocation_comparison_oos_source_digest(
        contract,
        scenario=scenario,
    )
    opening = result.folds[0].opening_state_digest
    closing = result.folds[-1].closing_state_digest
    if opening is None or closing is None:
        raise ValueError("cash control requires complete state-chain digests")
    reasons = result.stitched.diagnostics.termination_reasons
    return AllocationCashControlEvidence(
        contract_digest=contract.digest,
        scenario=scenario,
        control_policy_digest=policy_digest,
        carrier_recipe_digest=contract.direct_recipe_digest,
        oos_source_digest=source_digest,
        ledger_digest=allocation_continuous_ledger_summary_digest(
            result,
            source_digest=source_digest,
        ),
        execution_digest=allocation_continuous_execution_summary_digest(
            result,
            source_digest=source_digest,
        ),
        validity_evidence_digest=validity_evidence_digest,
        opening_state_digest=opening,
        closing_state_digest=closing,
        terminal_profit_rate=terminal_profit,
        max_drawdown=max_drawdown,
        validity=validity,
        coverage_complete=True,
        termination_reason=None if not reasons else ",".join(reasons),
    )


__all__ = [
    "AllocationCashControlEvidence",
    "allocation_cash_control_policy_digest",
    "allocation_cash_reference",
    "build_allocation_cash_control_evidence",
    "run_continuous_allocation_cash_control",
]
