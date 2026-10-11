"""Predeclared settled controls; software reports, never a Study.

This separate factor comparison preserves the one-forecast A/B/C and PPO seed
matrix. No summary-only loader may claim the attached receipts were validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trade_rl._validation import require_unique_non_empty
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.allocation_global_execution import (
    GlobalAllocationForecastControls,
)
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
    _plan,
    _same,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    TerminalAllocationExecutionPlan,
)

_ROLES = ("control", "treatment", "cash")
_NONRL_ROLES = ("nonrl", "cash")


def _terminal_fixed_context(plan: TerminalAllocationExecutionPlan) -> dict[str, Any]:
    return {
        "stop": plan.settlement_stop_index,
        "closing_clocks_ns": plan.closing_clocks_ns,
        "closing_source_digest": plan.closing_source_digest,
        "clock": plan.bound.clock.payload(),
        "objective": {
            k: v
            for k, v in plan.bound.objective.payload().items()
            if k != "deployment_recipe_digest"
        },
    }


def _receipt_row(receipt: TerminalAllocationExecutionReceipt) -> dict[str, Any]:
    raw = receipt.payload
    final = raw["final_book"]
    return {
        "terminal_plan_digest": receipt.expected_plan.digest,
        "receipt_digest": receipt.digest,
        "status": raw["status"],
        "settlement_complete": raw["settlement_complete"],
        "risk_eligible": raw["risk_eligible"],
        "settled_profit_rate": raw["settled_profit_rate"],
        "observed_profit_rate": raw["observed_profit_rate"],
        "final_equity": None if final is None else final["equity"],
        "maximum_drawdown": None if final is None else final["max_drawdown"],
        "total_cost": None if final is None else final["total_cost"],
        "funding_pnl": None if final is None else final["funding_pnl"],
        "borrow_cost": None if final is None else final["borrow_cost"],
    }


@dataclass(frozen=True, slots=True)
class TerminalAllocationForecastControls:
    """Three complete frozen arms and the original external forecast assignment."""

    forecast_controls: GlobalAllocationForecastControls
    control_plan: TerminalAllocationExecutionPlan
    treatment_plan: TerminalAllocationExecutionPlan
    cash_plan: TerminalAllocationExecutionPlan

    def __post_init__(self) -> None:
        controls = self.forecast_controls
        if type(controls) is not GlobalAllocationForecastControls:
            raise ValueError(
                "settled controls require the exact native forecast declaration"
            )
        controls.__post_init__()
        plans = self.plans
        for plan, native in zip(
            plans,
            (controls.control_plan, controls.treatment_plan, controls.cash_plan),
            strict=True,
        ):
            _plan(plan)
            _same(plan.native_plan.payload, native.payload)
        fixed = tuple(_terminal_fixed_context(p) for p in plans)
        _same(fixed[0], fixed[1])
        _same(fixed[0], fixed[2])

    @property
    def plans(self) -> tuple[TerminalAllocationExecutionPlan, ...]:
        return self.control_plan, self.treatment_plan, self.cash_plan

    @property
    def key(self) -> tuple[str, str]:
        common = self.control_plan.native_plan.payload["common"]
        return common["account_id"], common["scenario"]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "allocation_terminal_forecast_controls_v1",
            "forecast_controls": self.forecast_controls.payload,
            "plans": {
                role: plan.payload
                for role, plan in zip(_ROLES, self.plans, strict=True)
            },
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


@dataclass(frozen=True, slots=True)
class SettledForecastComparisonPlan:
    """Expected Cartesian independent-account/scenario roster, declared before runs."""

    controls: tuple[TerminalAllocationForecastControls, ...]
    accounts: tuple[str, ...]
    scenarios: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (self.accounts, self.scenarios):
            if type(field) is not tuple:
                raise ValueError("settled report rosters must be native tuples")
            require_unique_non_empty(field, field="settled report roster")
        if type(self.controls) is not tuple or not self.controls:
            raise ValueError("settled report requires a nonempty frozen control tuple")
        for group in self.controls:
            if type(group) is not TerminalAllocationForecastControls:
                raise ValueError("settled report requires exact frozen groups")
            group.__post_init__()
        keys = tuple(group.key for group in self.controls)
        expected = {(a, s) for a in self.accounts for s in self.scenarios}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(
                "settled declaration omits or duplicates an account/scenario group"
            )
        pins = [p.digest for group in self.controls for p in group.plans]
        if len(set(pins)) != len(pins):
            raise ValueError("settled report repeats a complete arm identity")

    @property
    def ordered_groups(self) -> tuple[TerminalAllocationForecastControls, ...]:
        groups = {group.key: group for group in self.controls}
        return tuple(groups[a, s] for a in self.accounts for s in self.scenarios)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "allocation_settled_forecast_comparison_plan_v1",
            "accounts": list(self.accounts),
            "scenarios": list(self.scenarios),
            "groups": [group.payload for group in self.ordered_groups],
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


@dataclass(frozen=True, slots=True)
class SettledForecastComparisonReport:
    """Retain the validated receipts; summaries alone are not evidence authority."""

    expected_plan: SettledForecastComparisonPlan
    receipts: tuple[TerminalAllocationExecutionReceipt, ...]

    def __post_init__(self) -> None:
        if type(self.expected_plan) is not SettledForecastComparisonPlan:
            raise ValueError(
                "settled report requires the separately declared full matrix"
            )
        self.expected_plan.__post_init__()
        expected = tuple(
            p for group in self.expected_plan.ordered_groups for p in group.plans
        )
        if type(self.receipts) is not tuple or len(self.receipts) != len(expected):
            raise ValueError(
                "settled report requires every declared role/account/scenario receipt"
            )
        for receipt, plan in zip(self.receipts, expected, strict=True):
            if type(receipt) is not TerminalAllocationExecutionReceipt:
                raise ValueError("settled report requires closed terminal receipts")
            receipt.__post_init__()
            _same(receipt.expected_plan.payload, plan.payload)

    @property
    def payload(self) -> dict[str, Any]:
        groups = []
        for index, declaration in enumerate(self.expected_plan.ordered_groups):
            roles = {}
            for offset, role in enumerate(_ROLES):
                receipt = self.receipts[index * 3 + offset]
                roles[role] = _receipt_row(receipt)
            complete = all(row["risk_eligible"] for row in roles.values())
            groups.append(
                {
                    "account_id": declaration.key[0],
                    "scenario": declaration.key[1],
                    "controls_digest": declaration.digest,
                    "roles": roles,
                    "comparison_complete": complete,
                    "treatment_minus_control": roles["treatment"]["settled_profit_rate"]
                    - roles["control"]["settled_profit_rate"]
                    if complete
                    else None,
                    "treatment_minus_cash": roles["treatment"]["settled_profit_rate"]
                    - roles["cash"]["settled_profit_rate"]
                    if complete
                    else None,
                }
            )
        return {
            "schema": "allocation_settled_forecast_comparison_report_v1",
            "plan": self.expected_plan.payload,
            "plan_digest": self.expected_plan.digest,
            "groups": groups,
            "comparison_complete": all(g["comparison_complete"] for g in groups),
            "claim_scope": "bounded_software_comparison_no_research_qualification",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


def build_settled_forecast_comparison(
    expected_plan: SettledForecastComparisonPlan, records: tuple[object, ...]
) -> SettledForecastComparisonReport:
    """Require exactly the separately declared arms; perform no executions."""
    if type(expected_plan) is not SettledForecastComparisonPlan:
        raise ValueError("settled comparison requires the exact external matrix")
    expected_plan.__post_init__()
    plans = {p.digest: p for group in expected_plan.ordered_groups for p in group.plans}
    if type(records) is not tuple or len(records) != len(plans):
        raise ValueError("settled comparison omits or adds a declared cell")
    receipts = {}
    for record in records:
        if type(record) is not dict or type(record.get("plan_digest")) is not str:
            raise ValueError(
                "settled comparison requires complete raw terminal observations"
            )
        pin = record["plan_digest"]
        if pin not in plans or pin in receipts:
            raise ValueError("settled comparison repeats or substitutes a foreign arm")
        receipts[pin] = TerminalAllocationExecutionReceipt.from_payload(
            record, expected_plan=plans[pin]
        )
    return SettledForecastComparisonReport(
        expected_plan,
        tuple(
            receipts[p.digest]
            for group in expected_plan.ordered_groups
            for p in group.plans
        ),
    )


@dataclass(frozen=True, slots=True)
class TerminalAllocationNonRLControls:
    """One standalone forecast/controller and cash with the same declared inputs.

    Equal declared source/cost pins are not historical source authentication or
    proof that forecast cost estimates equal native realized trading charges.
    """

    nonrl_plan: TerminalAllocationExecutionPlan
    cash_plan: TerminalAllocationExecutionPlan

    def __post_init__(self) -> None:
        for plan in self.plans:
            _plan(plan)
        raw = tuple(plan.native_plan.payload for plan in self.plans)
        for value, kind in zip(raw, _NONRL_ROLES, strict=True):
            cell = value["cell"]
            if (
                cell["kind"] != kind
                or cell["original_recipe"] != cell["runtime"]["recipe"]
            ):
                raise ValueError("NONRL/cash controls require standalone native roles")
        _same(raw[0]["common"], raw[1]["common"])
        nonrl, cash = (value["cell"] for value in raw)
        cash_recipe = cash["original_recipe"]
        residual_cash = cash_recipe | {
            "action": cash_recipe["action"] | {"mode": "residual"}
        }
        _same(nonrl["original_recipe"], residual_cash)
        _same(nonrl["runtime"]["cost_payload"], cash["runtime"]["cost_payload"])
        _same(
            nonrl["runtime"]["source_context_digest"],
            cash["runtime"]["source_context_digest"],
        )
        _same(
            _terminal_fixed_context(self.nonrl_plan),
            _terminal_fixed_context(self.cash_plan),
        )

    @property
    def plans(self) -> tuple[TerminalAllocationExecutionPlan, ...]:
        return self.nonrl_plan, self.cash_plan

    @property
    def key(self) -> tuple[str, str]:
        common = self.nonrl_plan.native_plan.payload["common"]
        return common["account_id"], common["scenario"]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "allocation_terminal_nonrl_controls_v1",
            "plans": {
                role: plan.payload
                for role, plan in zip(_NONRL_ROLES, self.plans, strict=True)
            },
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


@dataclass(frozen=True, slots=True)
class SettledNonRLComparisonPlan:
    """A separate two-arm matrix, frozen before any native account starts."""

    controls: tuple[TerminalAllocationNonRLControls, ...]
    accounts: tuple[str, ...]
    scenarios: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (self.accounts, self.scenarios):
            if type(field) is not tuple:
                raise ValueError("NONRL report rosters must be native tuples")
            require_unique_non_empty(field, field="NONRL report roster")
        if type(self.controls) is not tuple or not self.controls:
            raise ValueError("NONRL report requires a nonempty frozen control tuple")
        for group in self.controls:
            if type(group) is not TerminalAllocationNonRLControls:
                raise ValueError("NONRL report requires exact frozen pairs")
            group.__post_init__()
        keys = tuple(group.key for group in self.controls)
        expected = {(a, s) for a in self.accounts for s in self.scenarios}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(
                "NONRL declaration omits or duplicates an account/scenario"
            )
        pins = [p.digest for group in self.controls for p in group.plans]
        if len(set(pins)) != len(pins):
            raise ValueError("NONRL report repeats a complete arm identity")

    @property
    def ordered_groups(self) -> tuple[TerminalAllocationNonRLControls, ...]:
        groups = {group.key: group for group in self.controls}
        return tuple(groups[a, s] for a in self.accounts for s in self.scenarios)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": "allocation_settled_nonrl_comparison_plan_v1",
            "accounts": list(self.accounts),
            "scenarios": list(self.scenarios),
            "groups": [group.payload for group in self.ordered_groups],
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


@dataclass(frozen=True, slots=True)
class SettledNonRLComparisonReport:
    """Closed observations of both standalone roles; no selected survivors."""

    expected_plan: SettledNonRLComparisonPlan
    receipts: tuple[TerminalAllocationExecutionReceipt, ...]

    def __post_init__(self) -> None:
        if type(self.expected_plan) is not SettledNonRLComparisonPlan:
            raise ValueError("NONRL report requires the separately frozen full matrix")
        self.expected_plan.__post_init__()
        expected = tuple(
            p for group in self.expected_plan.ordered_groups for p in group.plans
        )
        if type(self.receipts) is not tuple or len(self.receipts) != len(expected):
            raise ValueError("NONRL report requires every declared terminal receipt")
        for receipt, plan in zip(self.receipts, expected, strict=True):
            if type(receipt) is not TerminalAllocationExecutionReceipt:
                raise ValueError("NONRL report requires closed terminal receipts")
            receipt.__post_init__()
            _same(receipt.expected_plan.payload, plan.payload)

    @property
    def payload(self) -> dict[str, Any]:
        groups = []
        for index, declaration in enumerate(self.expected_plan.ordered_groups):
            roles = {
                role: _receipt_row(self.receipts[index * 2 + offset])
                for offset, role in enumerate(_NONRL_ROLES)
            }
            complete = all(row["risk_eligible"] for row in roles.values())
            groups.append(
                {
                    "account_id": declaration.key[0],
                    "scenario": declaration.key[1],
                    "controls_digest": declaration.digest,
                    "roles": roles,
                    "comparison_complete": complete,
                    "nonrl_minus_cash": roles["nonrl"]["settled_profit_rate"]
                    - roles["cash"]["settled_profit_rate"]
                    if complete
                    else None,
                }
            )
        return {
            "schema": "allocation_settled_nonrl_comparison_report_v1",
            "plan": self.expected_plan.payload,
            "plan_digest": self.expected_plan.digest,
            "groups": groups,
            "comparison_complete": all(g["comparison_complete"] for g in groups),
            "claim_scope": "bounded_software_comparison_no_research_qualification",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.payload)


def build_settled_nonrl_comparison(
    expected_plan: SettledNonRLComparisonPlan, records: tuple[object, ...]
) -> SettledNonRLComparisonReport:
    """Close both raw observations under their own plans, without any execution."""
    if type(expected_plan) is not SettledNonRLComparisonPlan:
        raise ValueError("NONRL comparison requires the exact external matrix")
    expected_plan.__post_init__()
    plans = {p.digest: p for group in expected_plan.ordered_groups for p in group.plans}
    if type(records) is not tuple or len(records) != len(plans):
        raise ValueError("NONRL comparison omits or adds a declared cell")
    receipts = {}
    for record in records:
        if type(record) is not dict or type(record.get("plan_digest")) is not str:
            raise ValueError("NONRL comparison requires complete raw observations")
        pin = record["plan_digest"]
        if pin not in plans or pin in receipts:
            raise ValueError("NONRL comparison repeats or substitutes a foreign arm")
        receipts[pin] = TerminalAllocationExecutionReceipt.from_payload(
            record, expected_plan=plans[pin]
        )
    return SettledNonRLComparisonReport(
        expected_plan,
        tuple(
            receipts[p.digest]
            for group in expected_plan.ordered_groups
            for p in group.plans
        ),
    )


__all__ = [
    "TerminalAllocationForecastControls",
    "SettledForecastComparisonPlan",
    "SettledForecastComparisonReport",
    "build_settled_forecast_comparison",
    "TerminalAllocationNonRLControls",
    "SettledNonRLComparisonPlan",
    "SettledNonRLComparisonReport",
    "build_settled_nonrl_comparison",
]
