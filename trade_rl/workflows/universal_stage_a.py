"""Universal architecture-ablation adapter for the maintained Stage A gate.

This module deliberately does not reimplement Stage A statistics or sealed-test
selection.  It only proves that the four Universal candidates differ in the
predeclared architecture projection and then delegates candidate identity,
checkpoint closure, validation selection, and sealed-test authorization to the
maintained Stage A contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
)
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    architecture_spec,
)

_ARCHITECTURE_CONFIG_KEYS = frozenset(
    {
        "observation_encoder",
        "policy_actor_head",
        "policy_net_arch",
        "value_net_arch",
        "sequence_tcn_capacity",
        "sequence_d_model",
        "sequence_timeframe_attention_heads",
        "sequence_timeframe_attention_layers",
        "sequence_timeframe_ffn_multiplier",
        "sequence_dropout",
    }
)


@dataclass(frozen=True, slots=True)
class UniversalStageACandidate:
    """One Universal ablation candidate bound to an exact Stage A candidate."""

    architecture: UniversalArchitectureName
    stage_a_candidate: StageACandidate
    training_config: ResidualTrainingConfig

    def __post_init__(self) -> None:
        architecture = UniversalArchitectureName(self.architecture)
        object.__setattr__(self, "architecture", architecture)
        if self.stage_a_candidate.candidate_id != architecture.value:
            raise ValueError("Universal Stage A candidate ID must equal architecture name")
        spec = architecture_spec(architecture)
        expected_fields: tuple[tuple[str, object, object], ...] = (
            (
                "observation_encoder",
                self.training_config.observation_encoder,
                "hierarchical_sequence_v2",
            ),
            (
                "sequence_tcn_capacity",
                self.training_config.sequence_tcn_capacity,
                spec.sequence_tcn_capacity,
            ),
            ("sequence_d_model", self.training_config.sequence_d_model, spec.d_model),
            (
                "sequence_timeframe_attention_heads",
                self.training_config.sequence_timeframe_attention_heads,
                spec.timeframe_attention_heads,
            ),
            (
                "sequence_timeframe_attention_layers",
                self.training_config.sequence_timeframe_attention_layers,
                spec.timeframe_attention_layers,
            ),
            (
                "sequence_timeframe_ffn_multiplier",
                self.training_config.sequence_timeframe_ffn_multiplier,
                spec.timeframe_ffn_multiplier,
            ),
            ("sequence_dropout", self.training_config.sequence_dropout, spec.sequence_dropout),
            ("policy_actor_head", self.training_config.policy_actor_head, spec.actor_head),
            ("policy_net_arch", self.training_config.policy_net_arch, spec.actor_mlp),
            ("value_net_arch", self.training_config.value_net_arch, spec.critic_mlp),
        )
        mismatches = tuple(
            field_name
            for field_name, actual, expected in expected_fields
            if actual != expected
        )
        if mismatches:
            raise ValueError(
                "Universal Stage A architecture projection mismatch: "
                + ", ".join(mismatches)
            )

    @property
    def fixed_condition_digest(self) -> str:
        payload = dict(self.training_config.digest_payload())
        for key in _ARCHITECTURE_CONFIG_KEYS:
            payload.pop(key, None)
        return content_digest(
            {
                "schema_version": "universal_stage_a_fixed_conditions_v1",
                "training_config": payload,
            }
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": "universal_stage_a_candidate_v1",
                "architecture": self.architecture.value,
                "stage_a_candidate_digest": self.stage_a_candidate.digest,
                "training_config_digest": content_digest(
                    self.training_config.digest_payload()
                ),
                "fixed_condition_digest": self.fixed_condition_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class UniversalStageAPlan:
    """Exact four-candidate Universal ablation projected onto Stage A."""

    ablation_candidates: tuple[UniversalStageACandidate, ...]
    stage_a_plan: StageAZeroShotEvaluationPlan

    def __post_init__(self) -> None:
        expected_architectures = tuple(UniversalArchitectureName)
        by_architecture: dict[UniversalArchitectureName, UniversalStageACandidate] = {}
        for candidate in self.ablation_candidates:
            if candidate.architecture in by_architecture:
                raise ValueError("Universal Stage A architectures must be unique")
            by_architecture[candidate.architecture] = candidate
        if set(by_architecture) != set(expected_architectures):
            raise ValueError(
                "Universal Stage A requires exactly the four declared ablation candidates"
            )
        ordered = tuple(by_architecture[name] for name in expected_architectures)
        object.__setattr__(self, "ablation_candidates", ordered)

        fixed_digests = {candidate.fixed_condition_digest for candidate in ordered}
        if len(fixed_digests) != 1:
            raise ValueError(
                "Universal Stage A non-architecture conditions must be identical"
            )

        if any(candidate.training_config.seeds != self.stage_a_plan.seeds for candidate in ordered):
            raise ValueError("Universal Stage A training seeds must match Stage A seeds")

        for candidate in ordered:
            try:
                declared = self.stage_a_plan.candidate(candidate.architecture.value)
            except ValueError as error:
                raise ValueError(
                    "Universal Stage A candidate identity is missing from Stage A plan"
                ) from error
            if declared.digest != candidate.stage_a_candidate.digest:
                raise ValueError(
                    "Universal Stage A candidate identity differs from Stage A plan"
                )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.architecture.value for candidate in self.ablation_candidates)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": "universal_stage_a_plan_v1",
                "stage_a_plan_digest": self.stage_a_plan.digest,
                "candidate_digests": tuple(
                    candidate.digest for candidate in self.ablation_candidates
                ),
                "fixed_condition_digest": self.ablation_candidates[0].fixed_condition_digest,
            }
        )


__all__ = ["UniversalStageACandidate", "UniversalStageAPlan"]
