"""Strict decoding for training-environment artifacts used by Serving promotion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

from trade_rl.rl.actions import ActionSpec
from trade_rl.simulation.execution import ExecutionCostConfig

TRAINING_ENVIRONMENT_SCHEMA = "training_environment_v2"
_ACTION_SPEC_FIELDS = frozenset(field.name for field in fields(ActionSpec))
_EXECUTION_COST_FIELDS = frozenset(
    field.name for field in fields(ExecutionCostConfig)
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _load_payload(path: Path) -> Mapping[str, object]:
    payload = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        field="training environment",
    )
    if payload.get("schema_version") != TRAINING_ENVIRONMENT_SCHEMA:
        raise ValueError("unsupported training environment schema")
    return payload


def _strict_mapping(
    value: object,
    *,
    expected_fields: frozenset[str],
    field: str,
) -> dict[str, object]:
    raw = dict(_mapping(value, field=field))
    observed = set(raw)
    missing = sorted(expected_fields - observed)
    unknown = sorted(observed - expected_fields)
    if missing:
        raise ValueError(f"missing {field} fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown {field} fields: {', '.join(unknown)}")
    return raw


def load_training_action_spec(path: Path) -> ActionSpec:
    """Load the complete action identity without applying local defaults."""

    payload = _load_payload(path)
    raw = _strict_mapping(
        payload.get("action"),
        expected_fields=_ACTION_SPEC_FIELDS,
        field="action",
    )
    return ActionSpec(**cast(dict[str, Any], raw))


def load_training_execution_cost(path: Path) -> ExecutionCostConfig:
    """Load one complete execution-cost identity without applying local defaults."""

    payload = _load_payload(path)
    environment = _mapping(payload.get("environment"), field="environment")
    raw = _strict_mapping(
        environment.get("execution_cost"),
        expected_fields=_EXECUTION_COST_FIELDS,
        field="execution_cost",
    )
    fractions = raw["trigger_volume_fractions"]
    if not isinstance(fractions, (list, tuple)):
        raise ValueError("trigger_volume_fractions must be a list or tuple")
    raw["trigger_volume_fractions"] = tuple(fractions)
    return ExecutionCostConfig(**cast(dict[str, Any], raw))


__all__ = [
    "TRAINING_ENVIRONMENT_SCHEMA",
    "load_training_action_spec",
    "load_training_execution_cost",
]
