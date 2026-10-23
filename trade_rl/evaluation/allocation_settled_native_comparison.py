"""Pure complete-roster comparison of source-matched native CASH/NONRL captures.

This is a software control report, never historical authentication or a winner.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.rl_allocation.settled_comparison_source import (
    SETTLED_NATIVE_CONTROL_SOURCE_FIELDS,
    settled_native_control_source_material,
)
from trade_rl.evaluation.rl_allocation.settled_native_validation import (
    SettledNativeActorIntent,
    SettledNativeCapture,
    SettledNativeValidationContext,
    validate_settled_native_capture,
)
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)

_METHODS = ("CASH", "NONRL")
_GROUP_FIELDS = frozenset(
    "group_id CASH_context_digest NONRL_context_digest CASH_account_id "
    "NONRL_account_id common_material_digest common_material".split()
)
_REPORT_GROUP_FIELDS = frozenset(
    "group_id common_material_digest arms primary_terminal_difference "
    "primary_return_difference".split()
)
_REPORT_ARM_FIELDS = frozenset(
    "method account_id context_digest capture_digest source_digest actor_intents_digest "
    "attempted_actor_intents_digest attempted_actor_intent_count verification "
    "observed_frontier_index observed_terminal_equity observed_terminal_return "
    "observed_total_cost observed_max_drawdown observed_fill_count observed_turnover_total".split()
)
_VERIFICATION_FIELDS = frozenset(
    "schema context_digest capture_digest counts validated_prefix returned_frontier "
    "frontier_coverage full_structural_settlement normal_phase_coverage dd_eligible "
    "admitted failure observed_equity_delta limitations".split()
)
_LIMITATIONS = (
    "G2_SOFTWARE_CONTROL_ONLY",
    "NO_RUNTIME_AUTHENTICATION",
    "OPAQUE_HISTORY_RNG",
    "NO_MARKET_PROFIT_OR_LEARNED_CONSUMPTION_AUTHORITY",
)


def _decode(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise ValueError("matched control artifacts require immutable bytes")
    value = json.loads(raw)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError("matched control artifacts require canonical JSON objects")
    return value


def _name(value: Any) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("matched control identity must be a nonempty string")


def _digest(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("matched control digest must be lowercase SHA256")


def _hash(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _number_or_none(value: Any) -> None:
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value)
    ):
        raise ValueError(
            "matched control metrics require finite native numbers or null"
        )


def _count(value: Any, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not int or value < 0:
        raise ValueError("matched control counts require nonnegative native integers")


@dataclass(frozen=True, slots=True)
class SettledNativeControlGroup:
    group_id: str
    cash_context: SettledNativeValidationContext
    nonrl_context: SettledNativeValidationContext


@dataclass(frozen=True, slots=True)
class SettledNativeControlComparisonPlan:
    """Pinned detached declaration; live sources are rechecked by the consumer."""

    _bytes: bytes

    def __post_init__(self) -> None:
        value = self.payload()
        if (
            set(value) != {"schema", "groups"}
            or value["schema"] != "allocation_settled_native_control_plan_v1"
            or type(value["groups"]) is not list
            or not value["groups"]
        ):
            raise ValueError("invalid matched control plan schema")
        groups, accounts = set(), set()
        for group in value["groups"]:
            if type(group) is not dict or set(group) != _GROUP_FIELDS:
                raise ValueError("invalid matched control plan group")
            _name(group["group_id"])
            if group["group_id"] in groups:
                raise ValueError("duplicate matched control group")
            groups.add(group["group_id"])
            common = group["common_material"]
            if (
                type(common) is not dict
                or set(common) != SETTLED_NATIVE_CONTROL_SOURCE_FIELDS
            ):
                raise ValueError("invalid matched control common source schema")
            _digest(group["common_material_digest"])
            if _hash(common) != group["common_material_digest"]:
                raise ValueError("matched control common material digest differs")
            for method in _METHODS:
                account = group[method + "_account_id"]
                _name(account)
                _digest(group[method + "_context_digest"])
                if account in accounts:
                    raise ValueError(
                        "matched control accounts must be globally distinct"
                    )
                accounts.add(account)

    def payload(self) -> dict[str, Any]:
        return _decode(self._bytes)

    def to_bytes(self) -> bytes:
        return self._bytes

    @property
    def digest(self) -> str:
        return sha256(self._bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class SettledNativeControlArm:
    group_id: str
    method: str
    context: SettledNativeValidationContext
    capture: SettledNativeCapture
    planned_actor_intents: tuple[SettledNativeActorIntent, ...]


@dataclass(frozen=True, slots=True)
class SettledNativeControlComparisonReport:
    """Detached closed report grammar, never an admission input or credential."""

    _bytes: bytes

    def __post_init__(self) -> None:
        value = self.payload()
        if (
            set(value) != {"schema", "plan_digest", "groups", "limitations"}
            or value["schema"] != "allocation_settled_native_control_report_v1"
            or type(value["groups"]) is not list
            or not value["groups"]
            or value["limitations"] != list(_LIMITATIONS)
        ):
            raise ValueError("invalid matched control report schema")
        _digest(value["plan_digest"])
        groups, accounts = set(), set()
        for group in value["groups"]:
            if type(group) is not dict or set(group) != _REPORT_GROUP_FIELDS:
                raise ValueError("invalid matched control report group")
            _name(group["group_id"])
            _digest(group["common_material_digest"])
            if group["group_id"] in groups:
                raise ValueError("duplicate matched control report group")
            groups.add(group["group_id"])
            arms = group["arms"]
            if type(arms) is not list or len(arms) != 2:
                raise ValueError("matched control report requires both arms")
            for method, arm in zip(_METHODS, arms, strict=True):
                if (
                    type(arm) is not dict
                    or set(arm) != _REPORT_ARM_FIELDS
                    or arm["method"] != method
                ):
                    raise ValueError("invalid matched control report arm")
                _name(arm["account_id"])
                if arm["account_id"] in accounts:
                    raise ValueError("duplicate matched control report account")
                accounts.add(arm["account_id"])
                for field in (
                    "context_digest",
                    "capture_digest",
                    "source_digest",
                    "actor_intents_digest",
                    "attempted_actor_intents_digest",
                ):
                    _digest(arm[field])
                _count(arm["attempted_actor_intent_count"])
                _count(arm["observed_frontier_index"], nullable=True)
                _count(arm["observed_fill_count"], nullable=True)
                for field in (
                    "observed_terminal_equity",
                    "observed_terminal_return",
                    "observed_total_cost",
                    "observed_max_drawdown",
                    "observed_turnover_total",
                ):
                    _number_or_none(arm[field])
                verification = arm["verification"]
                if (
                    type(verification) is not dict
                    or set(verification) != _VERIFICATION_FIELDS
                    or verification["schema"]
                    != "allocation_settled_native_verification_v1"
                    or type(verification["admitted"]) is not bool
                    or verification["context_digest"] != arm["context_digest"]
                    or verification["capture_digest"] != arm["capture_digest"]
                ):
                    raise ValueError("invalid derived native verification grammar")
            eligible = all(arm["verification"]["admitted"] for arm in arms)
            for field in ("primary_terminal_difference", "primary_return_difference"):
                _number_or_none(group[field])
            if (group["primary_terminal_difference"] is None) != (
                group["primary_return_difference"] is None
            ):
                raise ValueError("matched primary metrics must share nullness")
            if not eligible and (
                group["primary_terminal_difference"] is not None
                or group["primary_return_difference"] is not None
            ):
                raise ValueError(
                    "ineligible matched control report has primary metrics"
                )

    def payload(self) -> dict[str, Any]:
        return _decode(self._bytes)

    def to_bytes(self) -> bytes:
        return self._bytes


def declare_settled_native_control_comparison(
    groups: tuple[SettledNativeControlGroup, ...],
) -> SettledNativeControlComparisonPlan:
    if type(groups) is not tuple or not groups:
        raise ValueError("matched controls require a nonempty immutable group roster")
    rows = []
    for group in groups:
        if type(group) is not SettledNativeControlGroup:
            raise ValueError("matched controls require exact declared groups")
        row: dict[str, Any] = {"group_id": group.group_id}
        materials = []
        for method, context in zip(
            _METHODS, (group.cash_context, group.nonrl_context), strict=True
        ):
            material = settled_native_control_source_material(context)
            if (context.bound.clock.actor_steps, context.bound.clock.reserve_steps) != (
                2,
                1,
            ):
                raise ValueError("matched controls require the declared P2/R1/H3 clock")
            if context.action_contract.mode != (
                "direct" if method == "CASH" else "residual"
            ):
                raise ValueError(
                    "matched control method differs from its action contract"
                )
            row[method + "_context_digest"] = context.digest
            row[method + "_account_id"] = context.account_id
            materials.append(material)
        if canonical_json_bytes(materials[0]) != canonical_json_bytes(materials[1]):
            raise ValueError(
                "matched controls differ in fixed financial/source conditions"
            )
        row.update(
            common_material=materials[0], common_material_digest=_hash(materials[0])
        )
        rows.append(row)
    return SettledNativeControlComparisonPlan(
        canonical_json_bytes(
            {"schema": "allocation_settled_native_control_plan_v1", "groups": rows}
        )
    )


def _arm_record(arm: SettledNativeControlArm) -> dict[str, Any]:
    if (
        type(arm.capture) is not SettledNativeCapture
        or type(arm.planned_actor_intents) is not tuple
    ):
        raise ValueError(
            "matched controls require raw captures and complete planned intents"
        )
    action = 0 if arm.method == "CASH" else 2
    expected = tuple(
        SettledNativeActorIntent(i, arm.context.start_index + i, action)
        for i in range(arm.context.bound.clock.actor_steps)
    )
    if (
        any(
            type(i) is not SettledNativeActorIntent
            or any(type(v) is not int for v in (i.ordinal, i.start_index, i.action))
            for i in arm.planned_actor_intents
        )
        or arm.planned_actor_intents != expected
    ):
        raise ValueError(
            "matched control planned actions differ from its complete method schedule"
        )
    count = len(arm.capture.payload()["actor_calls"])
    if count > len(expected):
        raise ValueError("matched control attempted more than its planned actions")
    prefix = expected[:count]
    verification = validate_settled_native_capture(
        arm.context, arm.capture, actor_intents=prefix
    ).payload()
    frontier = verification["returned_frontier"]
    book = (
        frontier["native_account"]["book"]
        if frontier["encoding_error"] is None
        else None
    )
    terminal_equity = (
        book["equity"]
        if book is not None and frontier["index"] == arm.context.settlement_stop_index
        else None
    )
    return {
        "method": arm.method,
        "account_id": arm.context.account_id,
        "context_digest": arm.context.digest,
        "capture_digest": sha256(arm.capture.to_bytes()).hexdigest(),
        "source_digest": _hash(settled_training_source_receipt(arm.context)),
        "actor_intents_digest": _hash([asdict(value) for value in expected]),
        "attempted_actor_intents_digest": _hash([asdict(value) for value in prefix]),
        "attempted_actor_intent_count": count,
        "verification": verification,
        "observed_frontier_index": frontier["index"],
        "observed_terminal_equity": terminal_equity,
        "observed_terminal_return": (terminal_equity - arm.context.initial_capital)
        / arm.context.initial_capital
        if terminal_equity is not None
        else None,
        "observed_total_cost": book["total_cost"] if book is not None else None,
        "observed_max_drawdown": book["max_drawdown"] if book is not None else None,
        "observed_fill_count": book["fill_count"] if book is not None else None,
        "observed_turnover_total": book["turnover_total"] if book is not None else None,
    }


def build_settled_native_control_comparison(
    plan: SettledNativeControlComparisonPlan,
    arms: tuple[SettledNativeControlArm, ...],
) -> SettledNativeControlComparisonReport:
    if type(plan) is not SettledNativeControlComparisonPlan or type(arms) is not tuple:
        raise ValueError(
            "matched controls require their exact plan and immutable full roster"
        )
    declared = plan.payload()["groups"]
    if len(arms) != 2 * len(declared) or any(
        type(arm) is not SettledNativeControlArm for arm in arms
    ):
        raise ValueError("matched control arm roster is incomplete or foreign")
    groups = []
    for arm in arms:
        _name(arm.group_id)
        _name(arm.method)
    for ordinal, group in enumerate(declared):
        pair = arms[2 * ordinal : 2 * ordinal + 2]
        if tuple((arm.group_id, arm.method) for arm in pair) != tuple(
            (group["group_id"], method) for method in _METHODS
        ):
            raise ValueError(
                "matched control arm order differs from the predeclared roster"
            )
        groups.append(
            SettledNativeControlGroup(
                group["group_id"], pair[0].context, pair[1].context
            )
        )
    frozen_groups = tuple(groups)
    if (
        declare_settled_native_control_comparison(frozen_groups).to_bytes()
        != plan.to_bytes()
    ):
        raise ValueError("matched control context/source changed after declaration")
    rows = []
    for ordinal, group in enumerate(declared):
        pair = arms[2 * ordinal : 2 * ordinal + 2]
        records = [_arm_record(arm) for arm in pair]
        eligible = all(record["verification"]["admitted"] for record in records)
        difference = (
            records[1]["observed_terminal_equity"]
            - records[0]["observed_terminal_equity"]
            if eligible
            else None
        )
        rows.append(
            {
                "group_id": group["group_id"],
                "common_material_digest": group["common_material_digest"],
                "arms": records,
                "primary_terminal_difference": difference,
                "primary_return_difference": difference
                / pair[0].context.initial_capital
                if difference is not None
                else None,
            }
        )
    if (
        declare_settled_native_control_comparison(frozen_groups).to_bytes()
        != plan.to_bytes()
    ):
        raise ValueError("matched control source changed during comparison")
    return SettledNativeControlComparisonReport(
        canonical_json_bytes(
            {
                "schema": "allocation_settled_native_control_report_v1",
                "plan_digest": plan.digest,
                "groups": rows,
                "limitations": list(_LIMITATIONS),
            }
        )
    )


__all__ = [
    "SettledNativeControlGroup",
    "SettledNativeControlComparisonPlan",
    "SettledNativeControlArm",
    "SettledNativeControlComparisonReport",
    "declare_settled_native_control_comparison",
    "build_settled_native_control_comparison",
]
