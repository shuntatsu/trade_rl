"""Strict decoding for training-environment artifacts used by Serving promotion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

from trade_rl.simulation.execution import ExecutionCostConfig

TRAINING_ENVIRONMENT_SCHEMA = "training_environment_v2"


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def load_training_execution_cost(path: Path) -> ExecutionCostConfig:
    """Load one complete execution-cost identity without applying local defaults."""

    payload = _mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        field="training environment",
    )
    if payload.get("schema_version") != TRAINING_ENVIRONMENT_SCHEMA:
        raise ValueError("unsupported training environment schema")
    environment = _mapping(payload.get("environment"), field="environment")
    raw = dict(_mapping(environment.get("execution_cost"), field="execution_cost"))
    expected = {item.name for item in fields(ExecutionCostConfig)}
    observed = set(raw)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ValueError(f"missing execution_cost fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown execution_cost fields: {', '.join(unknown)}")
    fractions = raw["trigger_volume_fractions"]
    if not isinstance(fractions, (list, tuple)):
        raise ValueError("trigger_volume_fractions must be a list or tuple")
    raw["trigger_volume_fractions"] = tuple(fractions)
    try:
        return ExecutionCostConfig(**cast(dict[str, Any], raw))
    except TypeError as error:
        raise ValueError("execution_cost fields are invalid") from error


__all__ = ["TRAINING_ENVIRONMENT_SCHEMA", "load_training_execution_cost"]
