"""Pure, versioned declaration for a planned portfolio runtime execution scenario.

This contract records exact execution-policy inputs. It does not load models,
validate runtime environments, authorize evaluation, or execute a scenario.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Literal

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.simulation.execution import ExecutionCostConfig

PortfolioRuntimeScenarioRole = Literal["base", "required_stress", "diagnostic_stress"]

_SCHEMA = "portfolio_runtime_execution_scenario_v2"
_DIGEST_DOMAIN = b"trade_rl.portfolio_runtime_execution_scenario_v2\0"
_SCENARIO_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_ROLES = frozenset({"base", "required_stress", "diagnostic_stress"})
_ALLOWED_CHANGED_FIELDS = frozenset(
    {
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "impact_rate",
        "multiplier",
        "slippage_std",
        "tail_slippage_probability",
        "tail_slippage_multiplier",
        "order_latency_bars",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "borrow_rate_multiplier",
        "collateral_haircut",
        "fee_rate",
        "impact_rate",
        "limit_offset_rate",
        "lot_size",
        "maintenance_margin_rate",
        "maker_fee_rate",
        "max_leverage",
        "max_participation_rate",
        "minimum_notional",
        "multiplier",
        "slippage_std",
        "spread_rate",
        "tail_slippage_multiplier",
        "tail_slippage_probability",
        "taker_fee_rate",
        "tick_size",
    }
)
_INTEGER_FIELDS = frozenset({"order_latency_bars", "random_seed"})
_BOOLEAN_FIELDS = frozenset(
    {"allow_short", "partial_fill_carry", "processing_bar_volume_capacity"}
)
_STRING_FIELDS = frozenset({"margin_mode", "order_type", "path_mode"})
_POLICY_FIELDS = frozenset(
    {
        *_FLOAT_FIELDS,
        *_INTEGER_FIELDS,
        *_BOOLEAN_FIELDS,
        *_STRING_FIELDS,
        "schema_version",
        "trigger_volume_fractions",
    }
)


def _validate_sha256(value: object, *, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a lowercase SHA-256 digest")
    require_sha256(value, field=field)


def _execution_policy_payload(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not ExecutionCostConfig:
        raise TypeError(f"{field} must be exactly ExecutionCostConfig")
    payload = value.execution_policy_payload()
    if type(payload) is not dict or frozenset(payload) != _POLICY_FIELDS:
        raise ValueError(f"{field} execution policy schema is unsupported")
    for name in _FLOAT_FIELDS:
        item = getattr(value, name)
        if type(item) not in (int, float):
            raise TypeError(f"{field}.{name} must be a non-boolean number")
        if not math.isfinite(item):
            raise ValueError(f"{field}.{name} must be finite")
    for name in _INTEGER_FIELDS:
        if type(getattr(value, name)) is not int:
            raise TypeError(f"{field}.{name} must be an integer")
    for name in _BOOLEAN_FIELDS:
        if type(getattr(value, name)) is not bool:
            raise TypeError(f"{field}.{name} must be a boolean")
    for name in _STRING_FIELDS:
        if type(getattr(value, name)) is not str:
            raise TypeError(f"{field}.{name} must be a string")
    fractions = value.trigger_volume_fractions
    if type(fractions) is not tuple or len(fractions) != 4:
        raise TypeError(f"{field}.trigger_volume_fractions must be a four-value tuple")
    if any(type(item) not in (int, float) for item in fractions):
        raise TypeError(
            f"{field}.trigger_volume_fractions must contain non-boolean numbers"
        )
    if not all(math.isfinite(item) for item in fractions):
        raise ValueError(f"{field}.trigger_volume_fractions must be finite")
    if payload["schema_version"] != "execution_policy_v2":
        raise ValueError(f"{field} has an unsupported execution policy schema")
    canonical_json_bytes(payload)
    return payload


def _changed_fields(
    base: dict[str, object], runtime: dict[str, object]
) -> tuple[str, ...]:
    if base.keys() != runtime.keys():
        raise ValueError("base and runtime execution policy fields differ")
    return tuple(
        name
        for name in sorted(base)
        if canonical_json_bytes(base[name]) != canonical_json_bytes(runtime[name])
    )


@dataclass(frozen=True, slots=True)
class PortfolioRuntimeExecutionScenarioV2:
    """Typed scenario declaration with no execution or gate authority."""

    scenario_id: str
    role: PortfolioRuntimeScenarioRole
    scenario_plan_sha256: str
    training_context_sha256: str
    transform_digest: str
    checkpoint_sha256: str
    base_execution_cost: ExecutionCostConfig
    runtime_execution_cost: ExecutionCostConfig
    changed_fields: tuple[str, ...]
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        self._validated_components()

    def _validated_components(
        self,
    ) -> tuple[dict[str, object], dict[str, object], tuple[str, ...]]:
        if type(self.schema_version) is not str or self.schema_version != _SCHEMA:
            raise ValueError("scenario schema_version is unsupported")
        if (
            type(self.scenario_id) is not str
            or _SCENARIO_ID.fullmatch(self.scenario_id) is None
        ):
            raise ValueError("scenario_id must be a canonical lowercase identifier")
        role_value: object = self.role
        if type(role_value) is not str or role_value not in _ROLES:
            raise ValueError("scenario role is unsupported")
        _validate_sha256(self.scenario_plan_sha256, field="scenario_plan_sha256")
        _validate_sha256(self.training_context_sha256, field="training_context_sha256")
        _validate_sha256(self.transform_digest, field="transform_digest")
        _validate_sha256(self.checkpoint_sha256, field="checkpoint_sha256")

        base = _execution_policy_payload(
            self.base_execution_cost, field="base_execution_cost"
        )
        runtime = _execution_policy_payload(
            self.runtime_execution_cost, field="runtime_execution_cost"
        )
        actual = _changed_fields(base, runtime)
        unsupported = set(actual) - _ALLOWED_CHANGED_FIELDS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported execution field changes: {names}")
        if type(self.changed_fields) is not tuple or any(
            type(name) is not str for name in self.changed_fields
        ):
            raise TypeError("changed_fields must be a tuple of field names")
        if tuple(sorted(set(self.changed_fields))) != self.changed_fields:
            raise ValueError("changed_fields must be sorted and unique")
        if set(self.changed_fields) - _ALLOWED_CHANGED_FIELDS:
            raise ValueError("changed_fields contains an unsupported execution field")
        if self.changed_fields != actual:
            raise ValueError(
                "changed_fields must exactly match the canonical config diff"
            )
        if self.role == "base" and actual:
            raise ValueError("base scenario cannot change execution fields")
        if self.role != "base" and not actual:
            raise ValueError("stress scenario must change a supported execution field")
        return base, runtime, actual

    @property
    def actual_changed_fields(self) -> tuple[str, ...]:
        return self._validated_components()[2]

    @property
    def base_execution_cost_digest(self) -> str:
        self._validated_components()
        return self.base_execution_cost.execution_policy_digest

    @property
    def runtime_execution_cost_digest(self) -> str:
        self._validated_components()
        return self.runtime_execution_cost.execution_policy_digest

    def to_payload(self) -> dict[str, object]:
        base, runtime, actual = self._validated_components()
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "role": self.role,
            "scenario_plan_sha256": self.scenario_plan_sha256,
            "training_context_sha256": self.training_context_sha256,
            "transform_digest": self.transform_digest,
            "checkpoint_sha256": self.checkpoint_sha256,
            "base_execution_cost": base,
            "base_execution_cost_digest": self.base_execution_cost.execution_policy_digest,
            "runtime_execution_cost": runtime,
            "runtime_execution_cost_digest": self.runtime_execution_cost.execution_policy_digest,
            "changed_fields": list(self.changed_fields),
            "actual_changed_fields": list(actual),
            "scope": "declaration_only_no_load_fit_replay_or_authorization",
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(_DIGEST_DOMAIN + self.to_bytes()).hexdigest()


def verify_portfolio_runtime_execution_scenario_v2_bytes(
    raw: bytes,
    *,
    expected: PortfolioRuntimeExecutionScenarioV2,
    expected_digest: str,
) -> PortfolioRuntimeExecutionScenarioV2:
    """Verify exact canonical bytes against an external typed declaration/pin.

    Serialized input is never decoded into an authority-bearing object. The
    caller retains the expected typed declaration and its independent digest.
    """

    if type(raw) is not bytes:
        raise TypeError("raw must be bytes")
    if type(expected) is not PortfolioRuntimeExecutionScenarioV2:
        raise TypeError("expected must be PortfolioRuntimeExecutionScenarioV2")
    _validate_sha256(expected_digest, field="expected_digest")
    if expected.digest != expected_digest:
        raise ValueError("expected digest differs from typed declaration")
    if raw != expected.to_bytes():
        raise ValueError("scenario bytes differ from expected declaration")
    return expected


__all__ = [
    "PortfolioRuntimeExecutionScenarioV2",
    "PortfolioRuntimeScenarioRole",
    "verify_portfolio_runtime_execution_scenario_v2_bytes",
]
