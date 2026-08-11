"""U5 training orchestration for the four Universal architecture candidates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trade_rl.domain.common import require_sha256
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)
from trade_rl.workflows.universal_stage_a import (
    UniversalStageACandidate,
    build_universal_stage_a_candidate_from_training,
)
from trade_rl.workflows.universal_teacher_runtime import build_universal_oracle_batches
from trade_rl.workflows.universal_training_runner import (
    UniversalTrainingRuntime,
    assemble_universal_sb3_training_backend,
    train_universal_seeds,
)

UniversalRuntimeFactory = Callable[
    [UniversalArchitectureName, ResidualTrainingConfig], UniversalTrainingRuntime
]


def _static_runtime_identity(runtime: UniversalTrainingRuntime) -> tuple[object, ...]:
    routed = runtime.routed_environment_factory
    return (
        runtime.train_symbols,
        runtime.catalog_digest,
        runtime.partition_digest,
        runtime.split_manifest_digest,
        runtime.feature_schema_digest,
        runtime.statistics_digest,
        runtime.instrument_context_schema_digest,
        tuple(binding.digest for binding in routed.bindings),
        routed.partition_digest,
        routed.run_seed,
        routed.max_cached_environments,
    )


def train_universal_stage_a_ablation(
    *,
    base_training: ResidualTrainingConfig,
    runtime_factory: UniversalRuntimeFactory,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    output_root: Path,
    verbose: int = 0,
) -> tuple[UniversalStageACandidate, ...]:
    """Train exactly the declared U5 candidates under identical non-architecture conditions."""

    if not isinstance(base_training, ResidualTrainingConfig):
        raise TypeError("base_training must be ResidualTrainingConfig")
    if not callable(runtime_factory):
        raise TypeError("runtime_factory must be callable")
    require_sha256(normalizer_digest, field="U5 normalizer_digest")
    require_sha256(feature_schema_digest, field="U5 feature_schema_digest")
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
        raise ValueError("U5 verbose must be a non-negative integer")

    prepared: list[
        tuple[
            UniversalArchitectureName,
            ResidualTrainingConfig,
            UniversalTrainingRuntime,
        ]
    ] = []
    for architecture in tuple(UniversalArchitectureName):
        training = apply_architecture_to_training_config(base_training, architecture)
        runtime = runtime_factory(architecture, training)
        if not isinstance(runtime, UniversalTrainingRuntime):
            raise TypeError("runtime_factory must return UniversalTrainingRuntime")
        if runtime.feature_schema_digest != feature_schema_digest:
            raise ValueError("U5 runtime feature schema identity mismatch")
        prepared.append((architecture, training, runtime))

    static_identities = {
        _static_runtime_identity(runtime) for _, _, runtime in prepared
    }
    if len(static_identities) != 1:
        raise ValueError("U5 non-architecture runtime conditions must be identical")

    _, first_training, first_runtime = prepared[0]
    behavior_cloning_seed = first_training.behavior_cloning_seed
    if (
        isinstance(behavior_cloning_seed, bool)
        or not isinstance(behavior_cloning_seed, int)
        or not 0 <= behavior_cloning_seed <= 0xFFFFFFFF
    ):
        raise ValueError("U5 requires an explicit uint32 behavior_cloning_seed")
    shared_batches = build_universal_oracle_batches(
        train_symbols=first_runtime.train_symbols,
        bindings=first_runtime.routed_environment_factory.bindings,
        concrete_environment_factory=(
            first_runtime.routed_environment_factory.concrete_environment_factory
        ),
        fold_train_range=fold_train_range,
        behavior_cloning_seed=behavior_cloning_seed,
        n_envs=first_training.n_envs,
    )

    candidates: list[UniversalStageACandidate] = []
    for architecture, training, runtime in prepared:
        backend, bundle = assemble_universal_sb3_training_backend(
            routed_environment_factory=runtime.routed_environment_factory,
            training=training,
            fold_train_range=fold_train_range,
            normalizer_digest=normalizer_digest,
            feature_schema_digest=feature_schema_digest,
            oracle_batches=shared_batches,
            verbose=verbose,
        )
        bound_runtime = runtime.with_pretraining_artifact(
            bundle.teacher_artifact.artifact_digest
        )
        candidate_root = Path(output_root) / architecture.value
        manifest = train_universal_seeds(
            runtime=bound_runtime,
            training=training,
            backend=backend,
            output_root=candidate_root,
            architecture_name=architecture.value,
        )
        candidates.append(
            build_universal_stage_a_candidate_from_training(
                architecture=architecture,
                training_config=training,
                training_manifest=manifest,
                output_root=candidate_root,
            )
        )

    result = tuple(candidates)
    if tuple(item.architecture for item in result) != tuple(UniversalArchitectureName):
        raise RuntimeError("U5 architecture closure changed during training")
    if len({item.fixed_condition_digest for item in result}) != 1:
        raise ValueError("U5 non-architecture training conditions differ")
    return result


__all__ = ["UniversalRuntimeFactory", "train_universal_stage_a_ablation"]
