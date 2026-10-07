"""Pure completed native global evidence and explicitly scoped validity content.

The record binds separately declared content. It authenticates neither reviewer
independence nor native accounting, and authorizes no economic Study or execution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.allocation_cash_control import AllocationCashControlEvidence
from trade_rl.evaluation.allocation_comparison import (
    AllocationCandidateKind,
    AllocationComparisonContract,
    AllocationComparisonEvidence,
    AllocationValidity,
)
from trade_rl.evaluation.allocation_comparison_evidence import (
    allocation_comparison_oos_source_digest,
    allocation_comparison_scenario_digest,
    allocation_policy_schedule_digest,
    continuous_account_profit_and_drawdown,
)
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationExecutionPlan,
    GlobalAllocationExecutionReceipt,
    ObservedGlobalAllocationExecution,
)
from trade_rl.evaluation.allocation_scenario_identity import (
    allocation_candidate_recipe_digest,
)
from trade_rl.evaluation.rl_allocation.global_walk_forward import (
    GlobalAllocationWalkForwardResult,
)
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    StitchMode,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind

_SCOPES = {"bounded_software_g2", "independent_g3"}
_KINDS = {"nonrl", "residual_ppo", "direct_ppo", "cash"}
_VALIDITY_FIELDS = {
    "schema",
    "validity",
    "reasons",
    "assurance_scope",
    "contract_digest",
    "cell",
    "common_digest",
    "cell_plan_digest",
    "receipt_digest",
    "implementation_digest",
    "runtime_environment_digest",
    "review_evidence",
    "native_accounting_evidence",
}
_COUNTERS = {
    "turnover_total": "filled_turnover",
    "total_cost": "interval_cost",
    "funding_pnl": "interval_funding",
    "borrow_cost": "interval_borrow_cost",
    "fill_count": "fill_count",
    "rebalance_events": "rebalance_events",
}
_DIAGNOSTICS = {key: "n_trades" if key == "fill_count" else key for key in _COUNTERS}


def _closed(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} requires exactly its declared fields")
    return value


def _digest(value: object, field: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{field} requires a native SHA-256 string")
    require_sha256(value, field=field)


def _scope(value: object) -> str:
    if type(value) is not str or value not in _SCOPES:
        raise ValueError("validity assurance scope must be explicitly declared")
    return value


def _validity(value: object) -> dict[str, Any]:
    raw = _closed(value, _VALIDITY_FIELDS, "global validity record")
    if (
        type(raw["schema"]) is not str
        or raw["schema"] != "allocation_global_validity_record_v1"
    ):
        raise ValueError("unknown global validity schema")
    if type(raw["validity"]) is not str or raw["validity"] not in {"valid", "invalid"}:
        raise ValueError("global validity requires explicit VALID or INVALID")
    if type(raw["reasons"]) is not list or any(
        type(reason) is not str or not reason.strip() for reason in raw["reasons"]
    ):
        raise ValueError("validity reasons must be native nonempty strings")
    if raw["validity"] == "invalid" and not raw["reasons"]:
        raise ValueError("explicit INVALID requires reasons")
    _scope(raw["assurance_scope"])
    for field in _VALIDITY_FIELDS:
        if field.endswith("digest"):
            _digest(raw[field], field)
    cell = _closed(raw["cell"], {"kind", "seed", "scenario"}, "validity cell")
    if type(cell["kind"]) is not str or cell["kind"] not in _KINDS:
        raise ValueError("validity cell kind is not declared")
    if type(cell["scenario"]) is not str:
        raise ValueError("validity scenario must be a native string")
    require_non_empty(cell["scenario"], field="validity scenario")
    if cell["kind"] in {"nonrl", "cash"}:
        if cell["seed"] is not None:
            raise ValueError("validity control must be seedless")
    elif type(cell["seed"]) is not int or cell["seed"] < 0:
        raise ValueError("validity RL seed must be a native nonnegative integer")
    for field in ("review_evidence", "native_accounting_evidence"):
        reference = _closed(raw[field], {"record_id", "digest"}, field)
        if type(reference["record_id"]) is not str:
            raise ValueError("validity evidence reference requires a native record ID")
        require_non_empty(reference["record_id"], field=field)
        _digest(reference["digest"], field)
    return raw


@dataclass(frozen=True, slots=True)
class GlobalAllocationValidityRecord:
    """Detached scoped declaration; references are content pins, not approval."""

    _bytes: bytes

    def __post_init__(self) -> None:
        if type(self._bytes) is not bytes:
            raise ValueError("validity record requires detached canonical bytes")
        if canonical_json_bytes(_validity(json.loads(self._bytes))) != self._bytes:
            raise ValueError("global validity record is not canonical")

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._bytes)

    @property
    def digest(self) -> str:
        return content_digest(self.payload)

    @classmethod
    def from_payload(cls, value: object) -> GlobalAllocationValidityRecord:
        return cls(canonical_json_bytes(_validity(value)))


def _number(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or type(value) not in (int, float)
        or not math.isfinite(value)
    ):
        raise ValueError(f"native {field} must be finite numeric content")
    return float(value)


def _equal(actual: object, expected: object, field: str) -> None:
    if not math.isclose(
        _number(actual, field), _number(expected, field), rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(f"native {field} disagrees with account evidence")


def _bound_validity(
    contract: AllocationComparisonContract,
    expected_plan: GlobalAllocationExecutionPlan,
    receipt: GlobalAllocationExecutionReceipt,
    record: GlobalAllocationValidityRecord,
    expected_scope: str,
) -> AllocationValidity:
    scope = _scope(expected_scope)
    if type(record) is not GlobalAllocationValidityRecord:
        raise ValueError("global validity requires the exact separate record")
    record.__post_init__()
    raw = record.payload
    common, cell = expected_plan.payload["common"], expected_plan.payload["cell"]
    expected = {
        "assurance_scope": scope,
        "contract_digest": contract.digest,
        "cell": {
            "kind": cell["kind"],
            "seed": cell["seed"],
            "scenario": common["scenario"],
        },
        "common_digest": expected_plan.common_digest,
        "cell_plan_digest": expected_plan.cell_plan_digest,
        "receipt_digest": receipt.digest,
        "implementation_digest": common["implementation_digest"],
        "runtime_environment_digest": common["runtime_environment_digest"],
    }
    if canonical_json_bytes(
        {key: raw[key] for key in expected}
    ) != canonical_json_bytes(expected):
        raise ValueError("validity scope or context/receipt binding differs")
    return AllocationValidity(raw["validity"])


def _context(
    contract: AllocationComparisonContract, plan: GlobalAllocationExecutionPlan
) -> None:
    common, cell = plan.payload["common"], plan.payload["cell"]
    if contract.account_mode != "independent_symbol":
        raise ValueError("global contract requires independent_symbol account mode")
    for field in (
        "dataset_id",
        "objective_digest",
        "clock_digest",
        "forecast_context_digest",
        "fold_plan_digest",
        "initial_capital",
    ):
        if getattr(contract, field) != common[field]:
            raise ValueError(f"comparison contract {field} differs from global context")
    scenario = next(
        (value for value in contract.scenarios if value.name == common["scenario"]),
        None,
    )
    if scenario is None or scenario.digest != allocation_comparison_scenario_digest(
        name=common["scenario"],
        dataset_id=common["dataset_id"],
        forecast_context_digest=common["forecast_context_digest"],
        economics_digest=common["economics_digest"],
        risk_digest=common["risk_digest"],
    ):
        raise ValueError(
            "comparison scenario differs from actual global economics/risk"
        )
    if common["scenario"] == "base" and (
        contract.economics_digest != common["economics_digest"]
        or contract.risk_digest != common["risk_digest"]
    ):
        raise ValueError("base contract economics/risk differs from native context")
    kind = cell["kind"]
    candidate = (
        contract.nonrl_recipe_digest
        if kind == "nonrl"
        else contract.residual_recipe_digest
        if kind == "residual_ppo"
        else contract.direct_recipe_digest
    )
    if candidate != cell["original_candidate_digest"]:
        raise ValueError("registered candidate recipe differs from global cell")
    if allocation_candidate_recipe_digest(cell["runtime"]["recipe"]) != candidate:
        raise ValueError("actual runtime recipe differs from registered candidate")
    if kind in {"residual_ppo", "direct_ppo"} and cell["seed"] not in contract.rl_seeds:
        raise ValueError("global candidate policy seed is not registered")
    if cell["fee"] is not None:
        original = cell["original_recipe"]["runtime_profile"]
        if (
            cell["fee"]["contract_digest"] != contract.digest
            or original["economics_digest"] != contract.economics_digest
            or original["risk_digest"] != contract.risk_digest
        ):
            raise ValueError(
                "original fee contract/base runtime differs from comparison"
            )


def _native_metrics(
    raw: dict[str, Any], result: GlobalAllocationWalkForwardResult
) -> tuple[float, float]:
    common = raw["plan"]["common"]
    previous: dict[str, Any] = {
        "equity": common["initial_capital"],
        "peak_value": common["initial_capital"],
        "max_drawdown": 0.0,
        **{name: 0 for name in _COUNTERS},
    }
    before_segment = dict(previous)
    segment_index = 0
    symbol = common["symbol_index"]
    values: list[float] = []
    for offset, row in enumerate(raw["rows"]):
        native = row["native"]
        book, execution = native["book"], native["execution"]
        decision = native["proposal"]["decision"]
        context = decision["baseline"]["context"]
        _equal(context["equity"], previous["equity"], "opening interval equity")
        _equal(decision["max_drawdown"], previous["max_drawdown"], "decision drawdown")
        expected_weights = (
            [0.0] * len(book["quantities"])
            if offset == 0
            else [
                quantity * mark * multiplier / previous["equity"]
                for quantity, mark, multiplier in zip(
                    previous["quantities"],
                    previous["mark_prices"],
                    previous["contract_multipliers"],
                    strict=True,
                )
            ]
        )
        for weight, expected in zip(
            native["current_weights"], expected_weights, strict=True
        ):
            _equal(weight, expected, "pretrade current weight")
        _equal(
            context["current_weight"],
            expected_weights[symbol],
            "decision current weight",
        )
        equity, peak, dd = (
            _number(book[key], key) for key in ("equity", "peak_value", "max_drawdown")
        )
        if (
            equity <= 0
            or peak <= 0
            or dd < 0
            or peak + 1e-12 < max(previous["peak_value"], equity)
            or dd + 1e-12 < previous["max_drawdown"]
            or dd + 1e-12 < 1 - equity / peak
        ):
            raise ValueError("native cumulative equity/peak/drawdown is inconsistent")
        net_return = max(equity / previous["equity"] - 1, -1 + 1e-12)
        _equal(execution["interval_net_return"], net_return, "interval net return")
        _equal(
            execution["interval_log_return"],
            math.log1p(net_return),
            "interval log return",
        )
        values.append(net_return)
        for name, interval in _COUNTERS.items():
            actual = _number(book[name], name)
            if name != "funding_pnl" and (
                actual < 0 or actual + 1e-12 < previous[name]
            ):
                raise ValueError("native unsigned cumulative counter is negative")
            _equal(actual - previous[name], execution[interval], f"interval {name}")
        segment = result.folds[segment_index]
        if native["processing_index"] == segment.stop:
            diagnostics = segment.diagnostics.digest_payload()
            for name, diagnostic in _DIAGNOSTICS.items():
                _equal(
                    book[name] - before_segment[name],
                    diagnostics[diagnostic],
                    f"segment {name}",
                )
            before_segment = dict(book)
            segment_index += 1
        previous = book
    if segment_index != len(result.folds):
        raise ValueError("native segment diagnostics do not cover full global horizon")
    for name, diagnostic in _DIAGNOSTICS.items():
        _equal(
            previous[name],
            result.stitched.diagnostics.digest_payload()[diagnostic],
            f"stitched {name}",
        )
    for actual, expected in zip(result.stitched.returns.values, values, strict=True):
        _equal(actual, expected, "stitched native interval return")
    return_profit, return_dd = continuous_account_profit_and_drawdown(
        result.stitched.returns
    )
    terminal_profit = previous["equity"] / common["initial_capital"] - 1
    _equal(return_profit, terminal_profit, "geometric terminal profit")
    if return_dd > previous["max_drawdown"] + 1e-12:
        raise ValueError("return-path DD exceeds native cumulative drawdown")
    return terminal_profit, previous["max_drawdown"]


def _validated(
    contract: AllocationComparisonContract,
    observed: ObservedGlobalAllocationExecution,
    expected_plan: GlobalAllocationExecutionPlan,
    validity_record: GlobalAllocationValidityRecord,
    expected_scope: str,
    *,
    cash: bool,
) -> tuple[
    dict[str, Any], GlobalAllocationWalkForwardResult, AllocationValidity, float, float
]:
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("global evidence requires exact comparison contract")
    contract.__post_init__()
    if type(expected_plan) is not GlobalAllocationExecutionPlan:
        raise ValueError(
            "global evidence requires separately supplied exact expected plan"
        )
    expected_plan.__post_init__()
    if type(observed) is not ObservedGlobalAllocationExecution:
        raise ValueError("global evidence requires exact completed observed execution")
    observed.__post_init__()
    if type(observed.receipt) is not GlobalAllocationExecutionReceipt:
        raise ValueError("global evidence requires exact completed receipt")
    raw = observed.receipt.payload
    if canonical_json_bytes(raw["plan"]) != canonical_json_bytes(expected_plan.payload):
        raise ValueError(
            "observed global execution differs from the whole expected plan"
        )
    if (raw["plan"]["cell"]["kind"] == "cash") is not cash:
        raise ValueError("cash reference and candidate evidence are separate")
    _context(contract, expected_plan)
    if any(
        row["native"]["risk_config"]["drawdown_stop"] != contract.maximum_drawdown
        for row in raw["rows"]
    ):
        raise ValueError(
            "comparison drawdown guardrail differs from native objective/risk"
        )
    validity = _bound_validity(
        contract, expected_plan, observed.receipt, validity_record, expected_scope
    )
    native = observed.native_result
    result = (
        native
        if isinstance(native, GlobalAllocationWalkForwardResult)
        else native.walk_forward
    )
    reconstructed = stitch_oos(result.folds, mode=StitchMode.CONTINUOUS_ACCOUNT)
    if result.stitched != reconstructed or reconstructed.gaps:
        raise ValueError("supplied global stitched evidence differs from native folds")
    if reconstructed.returns.kind is not ReturnKind.DECISION_STEP:
        raise ValueError("global stitched return kind must be DECISION_STEP")
    if (
        len(reconstructed.returns.values) != raw["plan"]["common"]["horizon_steps"]
        or reconstructed.diagnostics.termination_reasons
    ):
        raise ValueError("global stitched evidence must cover complete solvent H")
    profit, drawdown = _native_metrics(raw, result)
    return raw, result, validity, profit, drawdown


def _digests(
    contract: AllocationComparisonContract,
    raw: dict[str, Any],
    observed: ObservedGlobalAllocationExecution,
    validity_record: GlobalAllocationValidityRecord,
) -> dict[str, str]:
    scenario = raw["plan"]["common"]["scenario"]
    source = content_digest(
        {
            "schema": "allocation_global_comparison_oos_source_v1",
            "legacy_oos_source_digest": allocation_comparison_oos_source_digest(
                contract, scenario=scenario
            ),
            "global_common_digest": raw["common_digest"],
        }
    )
    return {
        "oos_source_digest": source,
        "validity_evidence_digest": validity_record.digest,
        "opening_state_digest": raw["opening_state_digest"],
        "closing_state_digest": raw["closing_state_digest"],
        "ledger_digest": content_digest(
            {
                "schema": "allocation_global_native_ledger_evidence_v1",
                "oos_source_digest": source,
                "native_facts": [row["native"] for row in raw["rows"]],
                "segments": raw["segments"],
            }
        ),
        "execution_digest": content_digest(
            {
                "schema": "allocation_global_execution_evidence_v1",
                "oos_source_digest": source,
                "receipt_digest": observed.receipt.digest,
                "receipt": raw,
            }
        ),
    }


def build_global_allocation_comparison_evidence(
    contract: AllocationComparisonContract,
    observed: ObservedGlobalAllocationExecution,
    *,
    expected_plan: GlobalAllocationExecutionPlan,
    validity_record: GlobalAllocationValidityRecord,
    expected_assurance_scope: str,
) -> AllocationComparisonEvidence:
    """Consume complete candidate execution; no defaults or research authority."""
    raw, result, validity, profit, drawdown = _validated(
        contract,
        observed,
        expected_plan,
        validity_record,
        expected_assurance_scope,
        cash=False,
    )
    cell, common = raw["plan"]["cell"], raw["plan"]["common"]
    candidate = AllocationCandidateKind(cell["kind"])
    policy = (
        None
        if candidate is AllocationCandidateKind.NONRL
        else allocation_policy_schedule_digest(
            candidate, cell["seed"], result.policy_digests
        )
    )
    return AllocationComparisonEvidence(
        contract_digest=contract.digest,
        candidate=candidate,
        scenario=common["scenario"],
        seed=cell["seed"],
        policy_digest=policy,
        recipe_digest=cell["original_candidate_digest"],
        terminal_profit_rate=profit,
        max_drawdown=drawdown,
        validity=validity,
        coverage_complete=True,
        termination_reason=None,
        **_digests(contract, raw, observed, validity_record),
    )


def build_global_allocation_cash_control_evidence(
    contract: AllocationComparisonContract,
    observed: ObservedGlobalAllocationExecution,
    *,
    expected_plan: GlobalAllocationExecutionPlan,
    validity_record: GlobalAllocationValidityRecord,
    expected_assurance_scope: str,
) -> AllocationCashControlEvidence:
    """Consume native seedless HOLD0 cash, retaining actual configured interest."""
    raw, _, validity, profit, drawdown = _validated(
        contract,
        observed,
        expected_plan,
        validity_record,
        expected_assurance_scope,
        cash=True,
    )
    cell, common = raw["plan"]["cell"], raw["plan"]["common"]
    return AllocationCashControlEvidence(
        contract_digest=contract.digest,
        scenario=common["scenario"],
        control_policy_digest=cell["control_policy_digest"],
        carrier_recipe_digest=cell["original_candidate_digest"],
        terminal_profit_rate=profit,
        max_drawdown=drawdown,
        validity=validity,
        coverage_complete=True,
        termination_reason=None,
        **_digests(contract, raw, observed, validity_record),
    )


__all__ = [
    "GlobalAllocationValidityRecord",
    "build_global_allocation_comparison_evidence",
    "build_global_allocation_cash_control_evidence",
]
