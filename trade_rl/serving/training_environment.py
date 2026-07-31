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


def _strict_dataclass_mapping(
    value: object,
    *,
    dataclass_type: type[object],
    field: str,
) -> dict[str, object]:
    raw = dict(_mapping(value, field=field))
    expected = {item.name for item in fields(dataclass_type)}
    observed = set(raw)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ValueError(f"missing {field} fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown {field} fields: {', '.join(unknown)}")
    return raw


def load_training_action_spec(path: Path) -> ActionSpec:
    """Load the complete action identity without applying local defaults."""

    payload = _load_payload(path)
    raw = _strict_dataclass_mapping(
        payload.get("action"),
        dataclass_type=ActionSpec,
        field="action",
    )
    return ActionSpec(**cast(dict[str, Any], raw))


def load_training_execution_cost(path: Path) -> ExecutionCostConfig:
    """Load one complete execution-cost identity without applying local defaults."""

    payload = _load_payload(path)
    environment = _mapping(payload.get("environment"), field="environment")
    raw = _strict_dataclass_mapping(
        environment.get("execution_cost"),
        dataclass_type=ExecutionCostConfig,
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
