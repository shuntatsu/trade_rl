"""Validate declared PPO settings against one regular financial clock."""

from __future__ import annotations

import math
from typing import Any


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


def _positive_integer(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_allocation_clock(
    clock: object, ppo: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    values = _mapping(
        clock,
        {
            "schema",
            "decision_interval_seconds",
            "execution_interval_seconds",
            "reward_interval_seconds",
            "economic_horizon_seconds",
            "rollout_steps",
            "gamma",
            "gae_lambda",
            "reward_schema",
        },
        "clock",
    )
    if (
        values["schema"] != "financial_clock_v1"
        or values["reward_schema"] != "equity_delta_v1"
    ):
        raise ValueError("allocation clock must use fixed-capital equity increments")
    for name in (
        "decision_interval_seconds",
        "execution_interval_seconds",
        "reward_interval_seconds",
        "economic_horizon_seconds",
        "rollout_steps",
    ):
        _positive_integer(
            values[name], name, minimum=2 if name == "rollout_steps" else 1
        )
    decision = values["decision_interval_seconds"]
    if (
        values["execution_interval_seconds"] != decision
        or values["reward_interval_seconds"] != decision
    ):
        raise ValueError("allocation clock requires one execution bar per decision")
    if values["economic_horizon_seconds"] % decision:
        raise ValueError("allocation horizon must align with the decision clock")
    if (
        _finite(values["gamma"], "gamma") != 1.0
        or not 0 <= _finite(values["gae_lambda"], "gae_lambda") <= 1
    ):
        raise ValueError("allocation gamma must equal one and GAE be within [0, 1]")
    for key, clock_key in (
        ("gamma", "gamma"),
        ("gae_lambda", "gae_lambda"),
        ("n_steps", "rollout_steps"),
    ):
        if ppo[key] != values[clock_key]:
            raise ValueError("actual PPO settings differ from the financial clock")
    if any(
        values[key] != profile[key]
        for key in ("decision_interval_seconds", "economic_horizon_seconds")
    ):
        raise ValueError("clock differs from the runtime recipe")
    return values


def validate_ppo_training_budget(training: dict[str, Any]) -> dict[str, Any]:
    values = training
    ppo = _mapping(
        values["ppo"],
        {
            "gamma",
            "gae_lambda",
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "net_arch",
            "device",
        },
        "PPO settings",
    )
    if (
        _finite(ppo["gamma"], "PPO gamma") != 1.0
        or not 0 <= _finite(ppo["gae_lambda"], "PPO gae_lambda") <= 1
    ):
        raise ValueError("PPO gamma must equal one and GAE be within [0, 1]")
    _positive_integer(values["seed"], "seed", minimum=0)
    requested = _positive_integer(values["requested_timesteps"], "requested_timesteps")
    actual = _positive_integer(values["actual_timesteps"], "actual_timesteps")
    for name in ("n_steps", "batch_size", "n_epochs"):
        _positive_integer(
            ppo[name], name, minimum=2 if name in ("n_steps", "batch_size") else 1
        )
    if (
        actual != requested
        or requested % ppo["n_steps"]
        or ppo["n_steps"] % ppo["batch_size"]
    ):
        raise ValueError(
            "realized training budget must equal the requested whole rollout budget"
        )
    if ppo["device"] != "cpu" or _finite(ppo["learning_rate"], "learning_rate") <= 0:
        raise ValueError("allocation PPO must declare a positive CPU learning rate")
    architecture = ppo["net_arch"]
    if isinstance(architecture, dict):
        if set(architecture) != {"pi", "vf"}:
            raise ValueError("PPO architecture must declare exactly pi and vf networks")
        networks = tuple(architecture.values())
    else:
        networks = (architecture,)
    for network in networks:
        if not isinstance(network, list) or not network:
            raise ValueError(
                "PPO network architecture must be a nonempty integer array"
            )
        for width in network:
            _positive_integer(width, "network width")
    return ppo
