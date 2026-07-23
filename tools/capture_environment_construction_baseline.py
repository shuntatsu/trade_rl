from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path

import numpy as np

from tests.rl.test_target_weight_action import environment, target_spec


def normalize(value: object) -> object:
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, Enum):
        return normalize(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: normalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def book(env: object, name: str) -> dict[str, object]:
    value = getattr(env, name)
    return {
        "cash": float(value.cash),
        "quantities": normalize(value.quantities),
        "weights": normalize(value.weights),
        "portfolio_value": float(value.portfolio_value),
        "termination_reason": normalize(value.termination_reason),
    }


def main() -> None:
    env = environment(target_spec(count=2))
    payload: dict[str, object] = {
        "environment_digest": env.environment_digest,
        "action_spec_digest": env.action_spec_digest,
        "observation_contract_digest": env.observation_contract_digest,
        "execution_policy_digest": env.execution_policy_digest,
        "action_names": normalize(env.action_names),
        "action_space": {
            "shape": normalize(env.action_space.shape),
            "dtype": str(env.action_space.dtype),
            "low": normalize(env.action_space.low),
            "high": normalize(env.action_space.high),
        },
        "observation_space": {
            "type": type(env.observation_space).__name__,
            "shape": normalize(getattr(env.observation_space, "shape", None)),
            "dtype": str(getattr(env.observation_space, "dtype", None)),
        },
        "minimum_start_index": int(env._minimum_start_index),
        "nominal_episode_bars": int(env._nominal_episode_bars),
        "nominal_decision_bars": int(env._nominal_decision_bars),
        "resolved_decision_hours": float(env._resolved_decision_hours),
        "initial": {
            "index": int(env._index),
            "end_index": int(env._end_index),
            "hybrid": book(env, "hybrid"),
            "shadow": book(env, "shadow"),
            "previous_action": normalize(env._previous_action),
            "pending_hybrid_target": normalize(env._pending_hybrid_target),
            "pending_shadow_target": normalize(env._pending_shadow_target),
            "hybrid_order_book": normalize(env._hybrid_order_book),
            "shadow_order_book": normalize(env._shadow_order_book),
            "has_reset": bool(env._has_reset),
        },
    }
    reset_observation, reset_info = env.reset(
        seed=17,
        options={"start_idx": 10, "initial_state_mode": "cash"},
    )
    payload["reset"] = {
        "observation": normalize(reset_observation),
        "info": normalize(reset_info),
        "index": int(env._index),
        "end_index": int(env._end_index),
        "hybrid": book(env, "hybrid"),
        "shadow": book(env, "shadow"),
    }
    observation, reward, terminated, truncated, info = env.step(
        np.array([0.40, 0.0], dtype=np.float32)
    )
    payload["step"] = {
        "observation": normalize(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": normalize(info),
        "index": int(env._index),
        "hybrid": book(env, "hybrid"),
        "shadow": book(env, "shadow"),
        "previous_action": normalize(env._previous_action),
        "pending_hybrid_target": normalize(env._pending_hybrid_target),
        "pending_shadow_target": normalize(env._pending_shadow_target),
        "hybrid_order_book": normalize(env._hybrid_order_book),
        "shadow_order_book": normalize(env._shadow_order_book),
        "execution_state": normalize(env._execution_state),
        "action_diagnostics": normalize(env._action_diagnostics),
    }
    canonical = json.dumps(
        normalize(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    Path("environment-construction-baseline.json").write_text(
        json.dumps(
            {"sha256": digest, "payload": json.loads(canonical)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(digest)


if __name__ == "__main__":
    main()
