from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


replace_once(
    "trade_rl/rl/training.py",
    '''        sequence_active = encoder is ObservationEncoder.HIERARCHICAL_SEQUENCE_V2
        actor_head = self.policy_actor_head
        if actor_head is None:
            actor_head = (
                "hierarchical_gate_target_v1"
                if sequence_active
                else "standard_continuous_v1"
            )
        if not isinstance(actor_head, str):
            raise ValueError("policy_actor_head must be a string")
        expected_actor_head = (
            "hierarchical_gate_target_v1"
            if sequence_active
            else "standard_continuous_v1"
        )
        if actor_head != expected_actor_head:
            raise ValueError(
                f"policy_actor_head must be {expected_actor_head} for "
                f"observation_encoder={encoder}"
            )
        object.__setattr__(self, "policy_actor_head", actor_head)
        if (
            not math.isfinite(self.hierarchical_gate_temperature)
            or self.hierarchical_gate_temperature <= 0.0
        ):
            raise ValueError(
                "hierarchical_gate_temperature must be finite and positive"
            )
        if not sequence_active and self.hierarchical_gate_temperature != 1.0:
            raise ValueError(
                "hierarchical_gate_temperature is inactive for non-sequence actors"
            )
''',
    '''        sequence_active = encoder is ObservationEncoder.HIERARCHICAL_SEQUENCE_V2
        actor_head = self.policy_actor_head
        if actor_head is None:
            actor_head = (
                "hierarchical_gate_target_v1"
                if sequence_active
                else "standard_continuous_v1"
            )
        if not isinstance(actor_head, str):
            raise ValueError("policy_actor_head must be a string")
        sequence_actor_heads = {
            "hierarchical_gate_target_v1",
            "shared_target_v1",
        }
        if sequence_active:
            if actor_head not in sequence_actor_heads:
                raise ValueError(
                    "policy_actor_head must be hierarchical_gate_target_v1 or "
                    "shared_target_v1 for observation_encoder="
                    f"{encoder}"
                )
        elif actor_head != "standard_continuous_v1":
            raise ValueError(
                "policy_actor_head must be standard_continuous_v1 for "
                f"observation_encoder={encoder}"
            )
        object.__setattr__(self, "policy_actor_head", actor_head)
        if (
            not math.isfinite(self.hierarchical_gate_temperature)
            or self.hierarchical_gate_temperature <= 0.0
        ):
            raise ValueError(
                "hierarchical_gate_temperature must be finite and positive"
            )
        if actor_head == "shared_target_v1" and self.hierarchical_gate_temperature != 1.0:
            raise ValueError(
                "hierarchical_gate_temperature is inactive for "
                "policy_actor_head=shared_target_v1"
            )
        if not sequence_active and self.hierarchical_gate_temperature != 1.0:
            raise ValueError(
                "hierarchical_gate_temperature is inactive for non-sequence actors"
            )
''',
)

replace_once(
    "trade_rl/rl/policies.py",
    '''    def active_mask(self, actor_latent: torch.Tensor) -> torch.Tensor:
        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)
        return contexts[:, :, -1] > 0.5

    def forward(self, actor_latent: torch.Tensor) -> torch.Tensor:
''',
    '''    def active_mask(self, actor_latent: torch.Tensor) -> torch.Tensor:
        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)
        return contexts[:, :, -1] > 0.5

    def current_weights(self, actor_latent: torch.Tensor) -> torch.Tensor:
        contexts = actor_latent.reshape(-1, self.n_symbols, self.context_dim)
        active = contexts[:, :, -1] > 0.5
        return contexts[:, :, -2] * active.to(dtype=contexts.dtype)

    def forward(self, actor_latent: torch.Tensor) -> torch.Tensor:
''',
)

replace_once(
    "trade_rl/rl/policies.py",
    '''@dataclass(frozen=True, slots=True)
class HierarchicalActorOutputs:
''',
    '''@dataclass(frozen=True, slots=True)
class ActionStageOutputs:
    """Head-independent deterministic action stage for comparable telemetry."""

    current_weights: torch.Tensor
    deterministic_actions: torch.Tensor
    active_mask: torch.Tensor
    change_intensity: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class HierarchicalActorOutputs:
''',
)

replace_once(
    "trade_rl/rl/policies.py",
    '''    def deterministic_actions(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return distribution-free deterministic target weights for export."""

        features = self.extract_features(observations)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.mlp_extractor.forward_actor(features)
        return torch.tanh(self.action_net(latent_pi))

    def hierarchical_actor_outputs(
''',
    '''    def action_stage_outputs(
        self, observations: dict[str, torch.Tensor]
    ) -> ActionStageOutputs:
        """Return deterministic action stages without sampling exploration noise."""

        features = self.extract_features(observations)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.mlp_extractor.forward_actor(features)
        if isinstance(self.action_net, SharedPerAssetGateTargetHead):
            outputs = self.action_net.outputs(latent_pi)
            return ActionStageOutputs(
                current_weights=outputs.current_weights,
                deterministic_actions=outputs.composed_actions,
                active_mask=outputs.active_mask,
                change_intensity=outputs.change_intensity,
            )
        if not isinstance(self.action_net, SharedPerAssetActionHead):
            raise RuntimeError("policy action head does not expose action stages")
        active_mask = self.action_net.active_mask(latent_pi)
        deterministic = torch.tanh(self.action_net(latent_pi))
        deterministic = deterministic * active_mask.to(dtype=deterministic.dtype)
        return ActionStageOutputs(
            current_weights=self.action_net.current_weights(latent_pi),
            deterministic_actions=deterministic,
            active_mask=active_mask,
            change_intensity=None,
        )

    def deterministic_actions(
        self, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return distribution-free deterministic target weights for export."""

        return self.action_stage_outputs(observations).deterministic_actions

    def hierarchical_actor_outputs(
''',
)

replace_once(
    "trade_rl/rl/policies.py",
    '''__all__ = [
    "AssetSetFeatureExtractor",
    "HierarchicalActorOutputs",
''',
    '''__all__ = [
    "ActionStageOutputs",
    "AssetSetFeatureExtractor",
    "HierarchicalActorOutputs",
''',
)

replace_once(
    "trade_rl/rl/tensorboard_logging.py",
    '''                output_factory = getattr(
                    getattr(self.model, "policy", None),
                    "hierarchical_actor_outputs",
                    None,
                )
''',
    '''                policy = getattr(self.model, "policy", None)
                output_factory = getattr(policy, "action_stage_outputs", None)
                if not callable(output_factory):
                    output_factory = getattr(
                        policy,
                        "hierarchical_actor_outputs",
                        None,
                    )
''',
)

replace_once(
    "trade_rl/rl/tensorboard_logging.py",
    '''                    intensity = outputs.change_intensity.detach().cpu().numpy()
                    current = outputs.current_weights.detach().cpu().numpy()
                    deterministic = outputs.composed_actions.detach().cpu().numpy()
''',
    '''                    raw_intensity = getattr(outputs, "change_intensity", None)
                    intensity = (
                        None
                        if raw_intensity is None
                        else raw_intensity.detach().cpu().numpy()
                    )
                    current = outputs.current_weights.detach().cpu().numpy()
                    raw_deterministic = getattr(
                        outputs,
                        "deterministic_actions",
                        getattr(outputs, "composed_actions", None),
                    )
                    if raw_deterministic is None:
                        return True
                    deterministic = raw_deterministic.detach().cpu().numpy()
''',
)

replace_once(
    "trade_rl/rl/tensorboard_logging.py",
    '''                        self._extend("trade_rl/change_intensity_mean", intensity)
                        for index, (
''',
    '''                        if intensity is not None:
                            self._extend("trade_rl/change_intensity_mean", intensity)
                        for index, (
''',
)

policy_identity = r'''"""Canonical policy identity shared by SB3 training, checkpoints, and serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.observations import CURRENT_WEIGHT_SOURCE
from trade_rl.rl.sequence_architecture import sequence_architecture_identity
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

SB3_POLICY_IDENTITY_ATTRIBUTE: Final = "_trade_rl_policy_identity"
SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v4"
LEGACY_SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v2"
READABLE_LEGACY_SB3_POLICY_IDENTITY_SCHEMA: Final = "sb3_policy_identity_v3"
POLICY_ARCHITECTURE_SCHEMA: Final = "shared_target_weight_policy_v3"
LEGACY_POLICY_ARCHITECTURE_SCHEMA: Final = "hierarchical_gate_target_policy_v2"
HIERARCHICAL_EXPLORATION_SCHEMA: Final = "target_weight_exploration_v2"
LEGACY_HIERARCHICAL_EXPLORATION_SCHEMA: Final = "hierarchical_exploration_v1"
HIERARCHICAL_ACTION_DISTRIBUTION: Final = "masked_shared_squashed_diag_gaussian_v1"
HIERARCHICAL_EXPLORATION_COUPLING: Final = "post_composition_gate_independent_v1"
DIRECT_EXPLORATION_COUPLING: Final = "direct_target_mean_v1"
HIERARCHICAL_LOG_STD_PARAMETERIZATION: Final = "shared_scalar_v1"
HIERARCHICAL_ACTOR_HEAD: Final = "hierarchical_gate_target_v1"
DIRECT_ACTOR_HEAD: Final = "shared_target_v1"
SUPPORTED_SEQUENCE_ACTOR_HEADS: Final = frozenset(
    {HIERARCHICAL_ACTOR_HEAD, DIRECT_ACTOR_HEAD}
)
CURRENT_WEIGHT_KEY: Final = "current_weights"
_INTERNAL_ACTION_DISTRIBUTION_NAMES: Final = frozenset(
    {
        "masked_shared_squashed_diag_gaussian",
        HIERARCHICAL_ACTION_DISTRIBUTION,
    }
)


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty string tuple")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} must be unique")
    return value


def _positive_temperature(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def current_weight_observation_identity(n_symbols: int) -> dict[str, object]:
    if isinstance(n_symbols, bool) or not isinstance(n_symbols, int) or n_symbols <= 0:
        raise ValueError("current-weight symbol count must be a positive integer")
    return {
        "bounds": (-1.0, 1.0),
        "dtype": "float32",
        "key": CURRENT_WEIGHT_KEY,
        "observation_schema": SEQUENCE_OBSERVATION_SCHEMA,
        "shape": (n_symbols,),
        "source": CURRENT_WEIGHT_SOURCE,
    }


def _validated_current_weight_identity(
    value: object, *, n_symbols: int
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("current-weight observation identity is missing")
    observed = dict(value)
    expected = current_weight_observation_identity(n_symbols)
    normalized = {
        **observed,
        "bounds": tuple(observed.get("bounds", ())),
        "shape": tuple(observed.get("shape", ())),
    }
    if normalized != expected:
        raise ValueError("current-weight observation identity mismatch")
    return expected


def _exploration_payload(actor_head: str) -> dict[str, object]:
    if actor_head not in SUPPORTED_SEQUENCE_ACTOR_HEADS:
        raise ValueError("unsupported sequence actor-head identity")
    coupling = (
        HIERARCHICAL_EXPLORATION_COUPLING
        if actor_head == HIERARCHICAL_ACTOR_HEAD
        else DIRECT_EXPLORATION_COUPLING
    )
    return {
        "action_distribution": HIERARCHICAL_ACTION_DISTRIBUTION,
        "log_std_parameterization": HIERARCHICAL_LOG_STD_PARAMETERIZATION,
        "mean_coupling": coupling,
        "state_dependent_noise": False,
        "schema_version": HIERARCHICAL_EXPLORATION_SCHEMA,
        "squashing": "tanh",
    }


def _legacy_exploration_payload() -> dict[str, object]:
    return {
        "action_distribution": HIERARCHICAL_ACTION_DISTRIBUTION,
        "change_intensity_coupling": HIERARCHICAL_EXPLORATION_COUPLING,
        "log_std_parameterization": HIERARCHICAL_LOG_STD_PARAMETERIZATION,
        "state_dependent_noise": False,
        "schema_version": LEGACY_HIERARCHICAL_EXPLORATION_SCHEMA,
        "squashing": "tanh",
    }


def _validated_exploration(value: object, *, actor_head: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("target-weight exploration identity is missing")
    observed = dict(value)
    expected = _exploration_payload(actor_head)
    if observed != expected:
        raise ValueError("target-weight exploration identity mismatch")
    return expected


def _validated_legacy_exploration(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("hierarchical exploration identity is missing")
    observed = dict(value)
    expected = _legacy_exploration_payload()
    if observed != expected:
        raise ValueError("hierarchical exploration identity mismatch")
    return expected


def _actual_exploration(policy: object, *, actor_head: str) -> dict[str, object]:
    distribution = getattr(policy, "action_distribution_name", None)
    if distribution not in _INTERNAL_ACTION_DISTRIBUTION_NAMES:
        raise ValueError("target-weight action distribution identity mismatch")
    log_std = getattr(policy, "log_std", None)
    shape = tuple(getattr(log_std, "shape", ()))
    if shape != (1,):
        raise ValueError("target-weight actor requires shared scalar log_std")
    if getattr(policy, "use_sde", None) is not False:
        raise ValueError("target-weight actor does not support gSDE")
    return _exploration_payload(actor_head)


def _policy_architecture_payload(
    *,
    actor_head: str,
    gate_temperature: float | None,
    sequence_architecture_digest: str,
    current_weight_observation: Mapping[str, object],
    exploration_contract: Mapping[str, object],
    schema_version: str = POLICY_ARCHITECTURE_SCHEMA,
) -> dict[str, object]:
    return {
        "actor_head": actor_head,
        "current_weight_observation": dict(current_weight_observation),
        "exploration_contract": dict(exploration_contract),
        "gate_temperature": gate_temperature,
        "observation_encoder": "hierarchical_sequence_v2",
        "schema_version": schema_version,
        "sequence_architecture_digest": sequence_architecture_digest,
    }


def _validated_payload(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("SB3 policy identity must be a non-empty mapping")
    payload = dict(value)
    schema = payload.get("schema_version")
    if schema == LEGACY_SB3_POLICY_IDENTITY_SCHEMA:
        raise ValueError(
            f"migrate {LEGACY_SB3_POLICY_IDENTITY_SCHEMA} to "
            f"{SB3_POLICY_IDENTITY_SCHEMA}"
        )
    if schema not in {
        READABLE_LEGACY_SB3_POLICY_IDENTITY_SCHEMA,
        SB3_POLICY_IDENTITY_SCHEMA,
    }:
        raise ValueError("unsupported SB3 policy identity schema")
    encoder = payload.get("observation_encoder")
    if encoder not in {"flat_mlp", "asset_set", "hierarchical_sequence_v2"}:
        raise ValueError("SB3 policy identity observation encoder is invalid")
    if encoder == "hierarchical_sequence_v2":
        architecture = payload.get("sequence_architecture")
        sequence_digest = payload.get("sequence_architecture_digest")
        if not isinstance(architecture, Mapping) or not architecture:
            raise ValueError("sequence architecture identity is missing")
        architecture_payload = dict(architecture)
        if not isinstance(sequence_digest, str) or sequence_digest != content_digest(
            architecture_payload
        ):
            raise ValueError("sequence architecture identity digest mismatch")
        for tuple_field in ("action_names", "symbols"):
            raw_tuple = architecture_payload.get(tuple_field)
            if isinstance(raw_tuple, list):
                architecture_payload[tuple_field] = tuple(raw_tuple)
        payload["sequence_architecture"] = architecture_payload
        raw_symbols = architecture_payload.get("symbols")
        symbols = _string_tuple(
            tuple(raw_symbols)
            if isinstance(raw_symbols, list | tuple)
            else raw_symbols,
            field="sequence architecture symbols",
        )
        actor_head = payload.get("actor_head")
        if schema == READABLE_LEGACY_SB3_POLICY_IDENTITY_SCHEMA:
            if actor_head != HIERARCHICAL_ACTOR_HEAD:
                raise ValueError("legacy hierarchical actor-head identity mismatch")
            temperature: float | None = _positive_temperature(
                payload.get("gate_temperature"), field="gate_temperature"
            )
            exploration = _validated_legacy_exploration(
                payload.get("exploration_contract")
            )
            architecture_schema = LEGACY_POLICY_ARCHITECTURE_SCHEMA
        else:
            if actor_head not in SUPPORTED_SEQUENCE_ACTOR_HEADS:
                raise ValueError("sequence actor-head identity mismatch")
            if actor_head == HIERARCHICAL_ACTOR_HEAD:
                temperature = _positive_temperature(
                    payload.get("gate_temperature"), field="gate_temperature"
                )
            else:
                if payload.get("gate_temperature") is not None:
                    raise ValueError("direct actor gate temperature must be null")
                temperature = None
            exploration = _validated_exploration(
                payload.get("exploration_contract"), actor_head=actor_head
            )
            architecture_schema = POLICY_ARCHITECTURE_SCHEMA
        current_weight = _validated_current_weight_identity(
            payload.get("current_weight_observation"), n_symbols=len(symbols)
        )
        payload["current_weight_observation"] = current_weight
        payload["exploration_contract"] = exploration
        payload["gate_temperature"] = temperature
        architecture_contract = _policy_architecture_payload(
            actor_head=actor_head,
            gate_temperature=temperature,
            sequence_architecture_digest=sequence_digest,
            current_weight_observation=current_weight,
            exploration_contract=exploration,
            schema_version=architecture_schema,
        )
        architecture_digest = payload.get("policy_architecture_digest")
        if not isinstance(
            architecture_digest, str
        ) or architecture_digest != content_digest(architecture_contract):
            raise ValueError("policy architecture identity digest mismatch")
    elif any(
        key in payload
        for key in (
            "actor_head",
            "current_weight_observation",
            "exploration_contract",
            "gate_temperature",
            "policy_architecture_digest",
            "sequence_architecture",
            "sequence_architecture_digest",
        )
    ):
        raise ValueError("non-sequence policy cannot declare sequence architecture")
    canonical_json_bytes(payload)
    return payload


def _actual_sequence_architecture(model: object) -> SequencePolicyArchitecture:
    policy = getattr(model, "policy", None)
    extractor = getattr(policy, "features_extractor", None)
    asset_encoder = getattr(extractor, "asset_encoder", None)
    architecture = getattr(asset_encoder, "architecture", None)
    if not isinstance(architecture, SequencePolicyArchitecture):
        raise ValueError(
            "hierarchical sequence model does not expose its validated architecture"
        )
    return architecture


def bind_sb3_policy_identity(model: Any, assembly: object) -> dict[str, object]:
    """Bind the actual assembled encoder and actor identity to an SB3 model."""
    encoder = getattr(assembly, "observation_encoder", None)
    if encoder not in {"flat_mlp", "asset_set", "hierarchical_sequence_v2"}:
        raise ValueError("SB3 assembly observation encoder is invalid")
    payload: dict[str, object] = {
        "observation_encoder": encoder,
        "schema_version": SB3_POLICY_IDENTITY_SCHEMA,
    }
    if encoder == "hierarchical_sequence_v2":
        symbols = _string_tuple(
            getattr(assembly, "sequence_symbols", None), field="sequence_symbols"
        )
        action_names = _string_tuple(
            getattr(assembly, "sequence_action_names", None),
            field="sequence_action_names",
        )
        identity = sequence_architecture_identity(
            _actual_sequence_architecture(model),
            symbols=symbols,
            action_names=action_names,
        )
        policy = getattr(model, "policy", None)
        actual_head = getattr(policy, "shared_actor_head", None)
        expected_head = getattr(assembly, "policy_actor_head", None)
        if (
            actual_head not in SUPPORTED_SEQUENCE_ACTOR_HEADS
            or expected_head != actual_head
        ):
            raise ValueError("sequence actor-head assembly identity mismatch")
        actual_raw_temperature = _positive_temperature(
            getattr(policy, "shared_actor_gate_temperature", None),
            field="model gate temperature",
        )
        expected_raw_temperature = _positive_temperature(
            getattr(assembly, "hierarchical_gate_temperature", None),
            field="assembly gate temperature",
        )
        if not math.isclose(
            actual_raw_temperature,
            expected_raw_temperature,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("gate-temperature assembly identity mismatch")
        if actual_head == DIRECT_ACTOR_HEAD:
            if not math.isclose(
                actual_raw_temperature, 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("direct actor gate temperature must remain inactive")
            gate_temperature: float | None = None
        else:
            gate_temperature = actual_raw_temperature
        current_weight = current_weight_observation_identity(len(symbols))
        exploration = _actual_exploration(policy, actor_head=actual_head)
        policy_architecture = _policy_architecture_payload(
            actor_head=actual_head,
            gate_temperature=gate_temperature,
            sequence_architecture_digest=identity.digest,
            current_weight_observation=current_weight,
            exploration_contract=exploration,
        )
        payload.update(
            {
                "actor_head": actual_head,
                "current_weight_observation": current_weight,
                "exploration_contract": exploration,
                "gate_temperature": gate_temperature,
                "policy_architecture_digest": content_digest(policy_architecture),
                "sequence_architecture": identity.digest_payload(),
                "sequence_architecture_digest": identity.digest,
            }
        )
    resolved = _validated_payload(payload)
    setattr(model, SB3_POLICY_IDENTITY_ATTRIBUTE, resolved)
    return dict(resolved)


def validated_sb3_policy_identity(value: object) -> dict[str, object]:
    """Validate and copy one serialized policy identity payload."""
    return dict(_validated_payload(value))


def model_sb3_policy_identity(model: object) -> dict[str, object] | None:
    raw = getattr(model, SB3_POLICY_IDENTITY_ATTRIBUTE, None)
    if raw is None:
        return None
    return _validated_payload(raw)


def validate_model_sb3_policy_identity(
    model: object, expected: Mapping[str, object]
) -> None:
    expected_payload = _validated_payload(expected)
    observed = model_sb3_policy_identity(model)
    if observed != expected_payload:
        raise ValueError("SB3 policy architecture identity mismatch")


__all__ = [
    "CURRENT_WEIGHT_KEY",
    "DIRECT_ACTOR_HEAD",
    "DIRECT_EXPLORATION_COUPLING",
    "HIERARCHICAL_ACTION_DISTRIBUTION",
    "HIERARCHICAL_ACTOR_HEAD",
    "HIERARCHICAL_EXPLORATION_COUPLING",
    "HIERARCHICAL_EXPLORATION_SCHEMA",
    "HIERARCHICAL_LOG_STD_PARAMETERIZATION",
    "POLICY_ARCHITECTURE_SCHEMA",
    "SB3_POLICY_IDENTITY_ATTRIBUTE",
    "SB3_POLICY_IDENTITY_SCHEMA",
    "SUPPORTED_SEQUENCE_ACTOR_HEADS",
    "bind_sb3_policy_identity",
    "current_weight_observation_identity",
    "model_sb3_policy_identity",
    "validate_model_sb3_policy_identity",
    "validated_sb3_policy_identity",
]
'''
write("trade_rl/rl/policy_identity.py", policy_identity)

direct_bc = r'''"""Causal quality gate for direct target-weight behavior cloning."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from trade_rl.learning.evaluation import (
    BehaviorCloningGateEvaluation,
    BehaviorCloningGateGroup,
    BehaviorCloningGateMetric,
    BehaviorCloningGateThresholds,
    evaluate_behavior_cloning_gates,
)


def _relative_improvement(initial_mse: object, final_mse: object) -> float | None:
    values = (initial_mse, final_mse)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        return None
    initial = float(initial_mse)
    final = float(final_mse)
    return (initial - final) / max(initial, float(np.finfo(np.float64).eps))


def evaluate_direct_behavior_cloning_gates(
    *,
    initial_mse: object,
    final_mse: object,
    teacher_change_support: int,
    holdout: Any,
    thresholds: BehaviorCloningGateThresholds,
) -> BehaviorCloningGateEvaluation:
    """Require direct-head reconstruction plus the canonical causal holdout gate."""

    if (
        isinstance(teacher_change_support, bool)
        or not isinstance(teacher_change_support, int)
        or teacher_change_support < 0
    ):
        raise ValueError("teacher_change_support must be a non-negative integer")
    improvement = _relative_improvement(initial_mse, final_mse)
    minimum_support = thresholds.minimum_teacher_positive_support
    if teacher_change_support < minimum_support:
        status = "insufficient_support"
        reason = (
            "action_mse_relative_improvement has support "
            f"{teacher_change_support}; minimum required support is {minimum_support}"
        )
    elif improvement is None:
        status = "insufficient_support"
        reason = "action_mse_relative_improvement is unavailable"
    elif improvement >= thresholds.minimum_composed_loss_relative_improvement:
        status = "passed"
        reason = "action_mse_relative_improvement passed"
    else:
        status = "failed"
        reason = "action-MSE improvement is below the required threshold"
    teacher_metric = BehaviorCloningGateMetric(
        name="action_mse_relative_improvement",
        status=status,
        observed=improvement,
        comparison=">=",
        threshold=thresholds.minimum_composed_loss_relative_improvement,
        support=teacher_change_support,
        minimum_support=minimum_support,
        reason=reason,
    )
    synthetic_metrics = SimpleNamespace(
        positive_support=teacher_change_support,
        active_target_rmse=0.0,
        activity_ratio=1.0,
        gate_precision=1.0,
        predicted_positive_support=teacher_change_support,
        gate_recall=1.0,
        constant_action_collapse=False,
        all_hold_collapse=False,
        all_trade_collapse=False,
    )
    canonical = evaluate_behavior_cloning_gates(
        initial_composed_loss=float(initial_mse)
        if isinstance(initial_mse, int | float) and not isinstance(initial_mse, bool)
        else None,
        final_composed_loss=float(final_mse)
        if isinstance(final_mse, int | float) and not isinstance(final_mse, bool)
        else None,
        reconstruction_metrics=synthetic_metrics,
        holdout=holdout,
        thresholds=thresholds,
    )
    return BehaviorCloningGateEvaluation(
        teacher_reconstruction_gate=BehaviorCloningGateGroup(
            name="teacher_reconstruction_gate",
            metrics=(teacher_metric,),
        ),
        causal_non_collapse_gate=canonical.causal_non_collapse_gate,
    )


__all__ = ["evaluate_direct_behavior_cloning_gates"]
'''
write("trade_rl/learning/direct_bc_evaluation.py", direct_bc)

replace_once(
    "tests/learning/test_direct_behavior_cloning_gate.py",
    '''from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    BehaviorCloningGateThresholds,
    evaluate_direct_behavior_cloning_gates,
)
''',
    '''from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    BehaviorCloningGateThresholds,
)
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''from trade_rl.learning.episode_behavior_cloning import (
''',
    '''from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
from trade_rl.learning.episode_behavior_cloning import (
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''def _hierarchical_teacher_labels(
    *,
    policy: object,
    teacher_dataset: SupervisedPolicyDataset,
    config: object,
) -> HierarchicalTeacherLabels | None:
    if not callable(getattr(policy, "hierarchical_actor_outputs", None)):
        return None
    observations = teacher_dataset.observations
    if not isinstance(observations, Mapping):
        raise ValueError("hierarchical BC requires structured teacher observations")
    missing = {"active", "current_weights"} - set(observations)
    if missing:
        raise ValueError(
            "hierarchical BC teacher observations are missing "
            + ", ".join(sorted(missing))
        )
    change_threshold = _required_hierarchical_config(
        config, "behavior_cloning_gate_change_threshold"
    )
    return build_hierarchical_teacher_labels(
        teacher_targets=np.asarray(teacher_dataset.actions),
        current_weights=np.asarray(observations["current_weights"]),
        active_mask=np.asarray(observations["active"]) > 0.5,
        change_threshold=float(change_threshold),
        source_teacher_digest=teacher_dataset.action_digest,
    )
''',
    '''def _teacher_change_labels(
    *,
    teacher_dataset: SupervisedPolicyDataset,
    config: object,
) -> HierarchicalTeacherLabels | None:
    observations = teacher_dataset.observations
    if not isinstance(observations, Mapping):
        return None
    missing = {"active", "current_weights"} - set(observations)
    if missing:
        raise ValueError(
            "structured BC teacher observations are missing "
            + ", ".join(sorted(missing))
        )
    change_threshold = _required_hierarchical_config(
        config, "behavior_cloning_gate_change_threshold"
    )
    return build_hierarchical_teacher_labels(
        teacher_targets=np.asarray(teacher_dataset.actions),
        current_weights=np.asarray(observations["current_weights"]),
        active_mask=np.asarray(observations["active"]) > 0.5,
        change_threshold=float(change_threshold),
        source_teacher_digest=teacher_dataset.action_digest,
    )


def _hierarchical_teacher_labels(
    *,
    policy: object,
    teacher_dataset: SupervisedPolicyDataset,
    config: object,
) -> HierarchicalTeacherLabels | None:
    if not callable(getattr(policy, "hierarchical_actor_outputs", None)):
        return None
    labels = _teacher_change_labels(teacher_dataset=teacher_dataset, config=config)
    if labels is None:
        raise ValueError("hierarchical BC requires structured teacher observations")
    return labels
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''                    hierarchical_labels = _hierarchical_teacher_labels(
                        policy=model.policy,
                        teacher_dataset=teacher_dataset,
                        config=config,
                    )
''',
    '''                    teacher_change_labels = _teacher_change_labels(
                        teacher_dataset=teacher_dataset,
                        config=config,
                    )
                    hierarchical_labels = (
                        teacher_change_labels
                        if callable(
                            getattr(model.policy, "hierarchical_actor_outputs", None)
                        )
                        else None
                    )
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''                    if hierarchical_labels is not None:
                        gate_evaluation = _evaluate_hierarchical_behavior_cloning_gate(
                            cloning=cloning,
                            holdout=holdout_evaluation,
                            thresholds=_behavior_cloning_gate_thresholds(config),
                        )
                        gate_evaluation_digest = write_learning_evaluation(
                            output_path.parent / "behavior-cloning-gates.json",
                            gate_evaluation,
                        )
                        quality_passed = gate_evaluation.passed
''',
    '''                    if hierarchical_labels is not None:
                        gate_evaluation = _evaluate_hierarchical_behavior_cloning_gate(
                            cloning=cloning,
                            holdout=holdout_evaluation,
                            thresholds=_behavior_cloning_gate_thresholds(config),
                        )
                    elif teacher_kind == "oracle" and teacher_change_labels is not None:
                        gate_evaluation = evaluate_direct_behavior_cloning_gates(
                            initial_mse=cloning.initial_mse,
                            final_mse=cloning.final_mse,
                            teacher_change_support=(
                                teacher_change_labels.diagnostics.gate_positive_count
                            ),
                            holdout=holdout_evaluation,
                            thresholds=_behavior_cloning_gate_thresholds(config),
                        )
                    if gate_evaluation is not None:
                        gate_evaluation_digest = write_learning_evaluation(
                            output_path.parent / "behavior-cloning-gates.json",
                            gate_evaluation,
                        )
                        quality_passed = gate_evaluation.passed
''',
)

replace_once(
    "tests/integrations/test_sb3_policy_identity_v3.py",
    'assert payload["schema_version"] == "sb3_policy_identity_v3"',
    'assert payload["schema_version"] == "sb3_policy_identity_v4"',
)
replace_once(
    "tests/integrations/test_sb3_policy_identity_v3.py",
    '''        "change_intensity_coupling": "post_composition_gate_independent_v1",
        "log_std_parameterization": "shared_scalar_v1",
        "state_dependent_noise": False,
        "schema_version": "hierarchical_exploration_v1",
''',
    '''        "log_std_parameterization": "shared_scalar_v1",
        "mean_coupling": "post_composition_gate_independent_v1",
        "state_dependent_noise": False,
        "schema_version": "target_weight_exploration_v2",
''',
)

replace_once(
    "tests/rl/test_action_head_ablation.py",
    '''def test_policy_identity_v3_requires_explicit_migration() -> None:
    payload = bind_sb3_policy_identity(
        _identity_model("hierarchical_gate_target_v1"),
        _identity_assembly("hierarchical_gate_target_v1"),
    )
    legacy = {**payload, "schema_version": "sb3_policy_identity_v3"}

    with pytest.raises(ValueError, match="migrate sb3_policy_identity_v3"):
        validated_sb3_policy_identity(legacy)
''',
    '''def test_policy_identity_v4_round_trips_through_validation() -> None:
    payload = bind_sb3_policy_identity(
        _identity_model("hierarchical_gate_target_v1"),
        _identity_assembly("hierarchical_gate_target_v1"),
    )

    assert validated_sb3_policy_identity(payload) == payload
''',
)

source_training = json.loads(
    (ROOT / "examples/binance-multitimeframe/training-target-weight-growth-ppo.json").read_text(
        encoding="utf-8"
    )
)
for suffix, actor_head in (
    ("gate", "hierarchical_gate_target_v1"),
    ("direct", "shared_target_v1"),
):
    payload = json.loads(json.dumps(source_training))
    payload["training"]["policy_actor_head"] = actor_head
    write(
        f"examples/binance-multitimeframe/training-action-head-ablation-{suffix}.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )

walk_forward = json.loads(
    (
        ROOT
        / "examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json"
    ).read_text(encoding="utf-8")
)
walk_forward["candidates"] = [
    {
        "name": "target-weight-gate-head-ppo",
        "run_file": "training-action-head-ablation-gate.json",
    },
    {
        "name": "target-weight-direct-head-ppo",
        "run_file": "training-action-head-ablation-direct.json",
    },
]
write(
    "examples/binance-multitimeframe/walk-forward-action-head-ablation.json",
    json.dumps(walk_forward, indent=2, ensure_ascii=False) + "\n",
)

binance = ROOT / "docs/BINANCE.md"
binance_text = binance.read_text(encoding="utf-8")
section = '''

## Action-head ablation

The maintained Gate-versus-direct target-weight comparison is declared by
`examples/binance-multitimeframe/walk-forward-action-head-ablation.json`.
Its two `run_file` candidates are required by regression test to differ only in
`training.policy_actor_head`; folds, seeds, Oracle teacher, encoder, PPO, reward,
risk, execution costs, and stress scenarios remain identical.

Run the same market walk-forward command used for the other canonical profiles,
substituting `walk-forward-action-head-ablation.json` as the configuration file.
Compare net return, baseline uplift, maximum drawdown, turnover, cost, action-stage
L1 metrics, and seed dispersion. This experiment does not authorize live exchange
routing.
'''
if "## Action-head ablation" not in binance_text:
    binance.write_text(binance_text.rstrip() + section + "\n", encoding="utf-8")

replace_once(
    "docs/superpowers/specs/2026-07-30-action-head-ablation-design.md",
    '''Upgrade the canonical policy identity to `sb3_policy_identity_v4`. Versions v2 and v3 are rejected with an explicit migration error because v3 encoded only the hierarchical head.
''',
    '''Upgrade newly bound policies to `sb3_policy_identity_v4`. Version v2 is rejected. Existing v3 hierarchical identities remain readable for checkpoint and serving migration, but no new v3 identity is produced and the direct head is valid only under v4.
''',
)
replace_once(
    "docs/superpowers/plans/2026-07-30-action-head-ablation.md",
    '''- Rejects serialized v2 and v3 identities.
''',
    '''- Rejects serialized v2 identities and reads existing hierarchical v3 identities only for migration compatibility.
''',
)

print("action-head ablation patch applied")
