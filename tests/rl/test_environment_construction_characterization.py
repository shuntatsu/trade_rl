from __future__ import annotations

import dataclasses
import hashlib
import json
import runpy
from enum import Enum
from typing import Any, Callable, cast

import numpy as np

BASELINE_SHA256 = "9d6540b3e3d3616bbb41caff036c6ef37228af56506adb030229aead86b11de1"
_fixture = runpy.run_path("tests/rl/test_target_weight_action.py")
environment = cast(Callable[..., Any], _fixture["environment"])
target_spec = cast(Callable[..., Any], _fixture["target_spec"])


def _normalize(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _normalize(value.tolist())
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Enum):
        return _normalize(value.value)
    snapshot = getattr(value, "snapshot", None)
    if callable(snapshot):
        return _normalize(snapshot())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _book(env: object, name: str) -> dict[str, object]:
    value = getattr(env, name)
    return {
        "cash": float(value.cash),
        "quantities": _normalize(value.quantities),
        "weights": _normalize(value.weights),
        "portfolio_value": float(value.portfolio_value),
        "termination_reason": _normalize(value.termination_reason),
    }


def _index_state(env: object) -> dict[str, int]:
    return {
        "start_index": int(getattr(env, "start_index")),
        "current_index": int(getattr(env, "current_index")),
        "end_index": int(getattr(env, "end_index")),
    }


def _characterization_payload() -> dict[str, object]:
    env = environment(target_spec(count=2))
    payload: dict[str, object] = {
        "environment_digest": env.environment_digest,
        "action_spec_digest": env.action_spec_digest,
        "observation_contract_digest": env.observation_contract_digest,
        "execution_policy_digest": env.execution_policy_digest,
        "action_names": _normalize(env.action_names),
        "action_space": {
            "shape": _normalize(env.action_space.shape),
            "dtype": str(env.action_space.dtype),
            "low": _normalize(env.action_space.low),
            "high": _normalize(env.action_space.high),
        },
        "observation_space": {
            "type": type(env.observation_space).__name__,
            "shape": _normalize(getattr(env.observation_space, "shape", None)),
            "dtype": str(getattr(env.observation_space, "dtype", None)),
        },
        "minimum_start_index": int(env._minimum_start_index),
        "nominal_episode_bars": int(env._nominal_episode_bars),
        "nominal_decision_bars": int(env._nominal_decision_bars),
        "resolved_decision_hours": float(env._resolved_decision_hours),
        "initial": {
            **_index_state(env),
            "hybrid": _book(env, "hybrid"),
            "shadow": _book(env, "shadow"),
            "previous_action": _normalize(env._previous_action),
            "pending_hybrid_target": _normalize(env._pending_hybrid_target),
            "pending_shadow_target": _normalize(env._pending_shadow_target),
            "hybrid_order_book": _normalize(env._hybrid_order_book),
            "shadow_order_book": _normalize(env._shadow_order_book),
            "has_reset": bool(env._has_reset),
        },
    }
    reset_observation, reset_info = env.reset(
        seed=17,
        options={"start_idx": 10, "initial_state_mode": "cash"},
    )
    payload["reset"] = {
        "observation": _normalize(reset_observation),
        "info": _normalize(reset_info),
        **_index_state(env),
        "hybrid": _book(env, "hybrid"),
        "shadow": _book(env, "shadow"),
    }
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.40, 0.0], dtype=np.float32)
    )
    payload["step"] = {
        "observation": _normalize(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _normalize(info),
        **_index_state(env),
        "hybrid": _book(env, "hybrid"),
        "shadow": _book(env, "shadow"),
        "previous_action": _normalize(env._previous_action),
        "pending_hybrid_target": _normalize(env._pending_hybrid_target),
        "pending_shadow_target": _normalize(env._pending_shadow_target),
        "hybrid_order_book": _normalize(env._hybrid_order_book),
        "shadow_order_book": _normalize(env._shadow_order_book),
        "execution_state": _normalize(env._execution_state),
        "action_diagnostics": _normalize(env._action_diagnostics),
    }
    return payload


def test_environment_construction_matches_pre_refactor_baseline() -> None:
    canonical = json.dumps(
        _normalize(_characterization_payload()),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == BASELINE_SHA256
