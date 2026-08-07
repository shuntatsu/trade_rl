"""Public framework-neutral training-environment identity contract."""

from __future__ import annotations

import math
from typing import Any, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.observations import OBSERVATION_SCHEMA


class _TrainingEnvironmentConfig(Protocol):
    @property
    def decision_hours(self) -> float | None: ...


def _combined_normalizer_digest(unwrapped: Any) -> str | None:
    flat = getattr(getattr(unwrapped, "normalizer", None), "digest", None)
    sequence = getattr(getattr(unwrapped, "sequence_normalizer", None), "digest", None)
    if flat is None and sequence is None:
        return None
    if sequence is None:
        return str(flat)
    if flat is None:
        return str(sequence)
    require_sha256(str(flat), field="normalizer_digest")
    require_sha256(str(sequence), field="sequence_normalizer_digest")
    return content_digest(
        {
            "flat": flat,
            "schema_version": "policy_normalizer_bundle_v1",
            "sequence": sequence,
        }
    )


def training_environment_identity(environment: object) -> dict[str, Any]:
    """Return the validated model-shape and economic identity of an environment."""

    unwrapped: Any = getattr(environment, "unwrapped", environment)
    environment_digest = getattr(unwrapped, "environment_digest", None)
    initial_capital = getattr(unwrapped, "initial_capital", None)
    if not isinstance(environment_digest, str):
        raise ValueError("training environment must expose environment_digest")
    require_sha256(environment_digest, field="environment_digest")
    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, int | float)
        or not math.isfinite(float(initial_capital))
        or float(initial_capital) <= 0.0
    ):
        raise ValueError("training environment must expose positive initial_capital")
    action_space = getattr(environment, "action_space", None)
    observation_space = getattr(environment, "observation_space", None)
    action_shape = getattr(action_space, "shape", None)
    observation_shape = getattr(observation_space, "shape", None)
    if not action_shape or len(action_shape) != 1 or action_shape[0] <= 0:
        raise ValueError("training environment must expose a flat action space")
    if observation_shape and len(observation_shape) == 1 and observation_shape[0] > 0:
        observation_size = int(observation_shape[0])
    else:
        component_spaces = getattr(observation_space, "spaces", None)
        if not isinstance(component_spaces, dict) or not component_spaces:
            raise ValueError(
                "training environment must expose a flat or structured observation space"
            )
        observation_size = 0
        for component in component_spaces.values():
            shape = getattr(component, "shape", None)
            if not shape or any(int(width) <= 0 for width in shape):
                raise ValueError("structured observation component has invalid shape")
            component_size = 1
            for width in shape:
                component_size *= int(width)
            observation_size += component_size
    observation_schema = getattr(unwrapped, "observation_schema", OBSERVATION_SCHEMA)
    observation_contract_digest = getattr(
        unwrapped, "observation_contract_digest", None
    )
    if not isinstance(observation_schema, str) or not observation_schema:
        raise ValueError("training environment must expose observation_schema")
    if observation_contract_digest is not None:
        require_sha256(observation_contract_digest, field="observation_contract_digest")
    return {
        "environment_digest": environment_digest,
        "initial_capital": float(initial_capital),
        "action_size": int(action_shape[0]),
        "action_names": tuple(getattr(unwrapped, "action_names", ())),
        "action_spec_digest": getattr(unwrapped, "action_spec_digest", None),
        "observation_size": observation_size,
        "observation_schema": observation_schema,
        "observation_contract_digest": observation_contract_digest,
        "decision_hours": getattr(unwrapped, "decision_hours", None),
        "alpha_artifact_digest": getattr(unwrapped, "alpha_artifact_digest", None),
        "factor_artifact_digest": getattr(unwrapped, "factor_artifact_digest", None),
        "normalizer_digest": _combined_normalizer_digest(unwrapped),
    }


def validate_training_environment(
    identity: dict[str, Any],
    config: _TrainingEnvironmentConfig,
) -> None:
    """Validate an environment identity against one authored training config."""

    action_size = int(identity["action_size"])
    action_names = identity["action_names"]
    action_spec_digest = identity["action_spec_digest"]
    if not isinstance(action_names, tuple) or len(action_names) != action_size:
        raise ValueError("training environment must expose exact action_names")
    if not isinstance(action_spec_digest, str):
        raise ValueError("training environment must expose action_spec_digest")
    require_sha256(action_spec_digest, field="action_spec_digest")
    environment_decision_hours = identity["decision_hours"]
    if config.decision_hours is not None:
        if not isinstance(environment_decision_hours, int | float):
            raise ValueError("training environment must expose decision_hours")
        if not math.isclose(
            float(environment_decision_hours),
            config.decision_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("training decision_hours do not match the environment")


__all__ = ["training_environment_identity", "validate_training_environment"]
