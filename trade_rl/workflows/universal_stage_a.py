"""Universal architecture-ablation adapter for the maintained Stage A gate.

This module deliberately does not reimplement Stage A statistics or sealed-test
selection.  It only proves that the four Universal candidates differ in the
predeclared architecture projection and then delegates candidate identity,
checkpoint closure, validation selection, and sealed-test authorization to the
maintained Stage A contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
)
from trade_rl.rl.checkpointing import checkpoint_manifests
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
            raise ValueError(
                "Universal Stage A candidate ID must equal architecture name"
            )
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
                spec.tcn_capacity,
            ),
            ("sequence_d_model", self.training_config.sequence_d_model, spec.d_model),
            (
                "sequence_timeframe_attention_heads",
                self.training_config.sequence_timeframe_attention_heads,
                spec.attention_heads,
            ),
            (
                "sequence_timeframe_attention_layers",
                self.training_config.sequence_timeframe_attention_layers,
                spec.attention_layers,
            ),
            (
                "sequence_timeframe_ffn_multiplier",
                self.training_config.sequence_timeframe_ffn_multiplier,
                spec.ffn_multiplier,
            ),
            (
                "sequence_dropout",
                self.training_config.sequence_dropout,
                spec.sequence_dropout,
            ),
            (
                "policy_actor_head",
                self.training_config.policy_actor_head,
                spec.actor_head,
            ),
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


def build_universal_stage_a_candidate_from_training(
    *,
    architecture: UniversalArchitectureName | str,
    training_config: ResidualTrainingConfig,
    training_manifest: Mapping[str, object],
    output_root: str | Path,
) -> UniversalStageACandidate:
    """Bind one completed Universal training run to exact Stage A checkpoints."""

    resolved_architecture = UniversalArchitectureName(architecture)
    if not isinstance(training_config, ResidualTrainingConfig):
        raise TypeError(
            "Universal Stage A training_config must be ResidualTrainingConfig"
        )
    manifest = dict(training_manifest)
    if manifest.get("schema_version") != "universal_training_run_v1":
        raise ValueError("Universal Stage A training manifest schema mismatch")
    if manifest.get("architecture_name") != resolved_architecture.value:
        raise ValueError("Universal Stage A training architecture mismatch")

    config_digest = content_digest(training_config.digest_payload())
    if manifest.get("training_config_digest") != config_digest:
        raise ValueError("Universal Stage A training config digest mismatch")
    run_digest = manifest.get("run_digest")
    if not isinstance(run_digest, str):
        raise ValueError("Universal Stage A training run digest is unavailable")
    require_sha256(run_digest, field="Universal Stage A training run_digest")
    run_payload = {key: value for key, value in manifest.items() if key != "run_digest"}
    if run_digest != content_digest(run_payload):
        raise ValueError("Universal Stage A training run digest mismatch")

    raw_members = manifest.get("members")
    if not isinstance(raw_members, list | tuple):
        raise TypeError("Universal Stage A training members must be a sequence")
    members = tuple(raw_members)
    seeds = tuple(training_config.seeds)
    if len(members) != len(seeds):
        raise ValueError("Universal Stage A training seed closure mismatch")

    checkpoint_digests: list[tuple[int, str]] = []
    policy_architecture_digest: str | None = None
    root = Path(output_root)
    for expected_seed, raw_member in zip(seeds, members, strict=True):
        if not isinstance(raw_member, Mapping):
            raise TypeError("Universal Stage A training member must be a mapping")
        member = dict(raw_member)
        if member.get("seed") != expected_seed:
            raise ValueError("Universal Stage A training member seed mismatch")
        actual_timesteps = member.get("actual_timesteps")
        if (
            isinstance(actual_timesteps, bool)
            or not isinstance(actual_timesteps, int)
            or actual_timesteps < training_config.timesteps
        ):
            raise ValueError("Universal Stage A training member is incomplete")
        environment_digest = member.get("environment_digest")
        architecture_digest = member.get("architecture_digest")
        if not isinstance(environment_digest, str):
            raise ValueError(
                "Universal Stage A member environment digest is unavailable"
            )
        if not isinstance(architecture_digest, str):
            raise ValueError(
                "Universal Stage A member architecture digest is unavailable"
            )
        require_sha256(
            environment_digest,
            field="Universal Stage A member environment_digest",
        )
        require_sha256(
            architecture_digest,
            field="Universal Stage A member architecture_digest",
        )

        manifests = checkpoint_manifests(root / f"seed-{expected_seed}" / "checkpoints")
        final = tuple(
            item
            for item in manifests
            if item.seed == expected_seed
            and item.requested_timestep == training_config.timesteps
            and item.observed_timestep == actual_timesteps
        )
        if len(final) != 1:
            raise ValueError(
                "Universal Stage A requires exactly one final checkpoint per seed"
            )
        checkpoint = final[0]
        if checkpoint.algorithm != training_config.algorithm:
            raise ValueError("Universal Stage A final checkpoint algorithm mismatch")
        if checkpoint.training_config_digest != config_digest:
            raise ValueError("Universal Stage A final checkpoint config mismatch")
        if checkpoint.environment_digest != environment_digest:
            raise ValueError("Universal Stage A final checkpoint environment mismatch")
        identity = checkpoint.algorithm_identity
        if not isinstance(identity, Mapping):
            raise ValueError(
                "Universal Stage A final checkpoint policy identity is missing"
            )
        policy = identity.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError(
                "Universal Stage A final checkpoint policy identity is missing"
            )
        checkpoint_architecture = policy.get("policy_architecture_digest")
        if checkpoint_architecture != architecture_digest:
            raise ValueError("Universal Stage A policy architecture identity mismatch")
        if policy_architecture_digest is None:
            policy_architecture_digest = architecture_digest
        elif policy_architecture_digest != architecture_digest:
            raise ValueError(
                "Universal Stage A policy architecture differs across seeds"
            )
        require_sha256(checkpoint.digest, field="Universal Stage A checkpoint digest")
        checkpoint_digests.append((expected_seed, checkpoint.digest))

    if policy_architecture_digest is None:  # pragma: no cover - seeds are non-empty
        raise RuntimeError("Universal Stage A policy architecture identity disappeared")
    stage_a_candidate = StageACandidate.create(
        candidate_id=resolved_architecture.value,
        candidate_config_digest=config_digest,
        final_training_completion_digest=run_digest,
        policy_identity=policy_architecture_digest,
        checkpoint_digests=tuple(checkpoint_digests),
    )
    return UniversalStageACandidate(
        architecture=resolved_architecture,
        stage_a_candidate=stage_a_candidate,
        training_config=training_config,
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

        if any(
            candidate.training_config.seeds != self.stage_a_plan.seeds
            for candidate in ordered
        ):
            raise ValueError(
                "Universal Stage A training seeds must match Stage A seeds"
            )

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
        return tuple(
            candidate.architecture.value for candidate in self.ablation_candidates
        )

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": "universal_stage_a_plan_v1",
                "stage_a_plan_digest": self.stage_a_plan.digest,
                "candidate_digests": tuple(
                    candidate.digest for candidate in self.ablation_candidates
                ),
                "fixed_condition_digest": self.ablation_candidates[
                    0
                ].fixed_condition_digest,
            }
        )


__all__ = [
    "UniversalStageACandidate",
    "UniversalStageAPlan",
    "build_universal_stage_a_candidate_from_training",
]
