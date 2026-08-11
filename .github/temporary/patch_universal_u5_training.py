from pathlib import Path

runner_path = Path("trade_rl/workflows/universal_training_runner.py")
runner = runner_path.read_text()

runner = runner.replace(
    "from dataclasses import asdict, dataclass\n",
    "from dataclasses import asdict, dataclass, replace\n",
    1,
)
if "from trade_rl.domain.common import require_sha256\n" not in runner:
    anchor = "from trade_rl.data.contracts import (\n"
    if anchor not in runner:
        raise SystemExit("runner data-contract import anchor not found")
    runner = runner.replace(
        anchor,
        "from trade_rl.domain.common import require_sha256\n" + anchor,
        1,
    )
if "from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch\n" not in runner:
    anchor = "from trade_rl.integrations.universal_pretraining import (\n"
    if anchor not in runner:
        raise SystemExit("runner universal-pretraining import anchor not found")
    runner = runner.replace(
        anchor,
        "from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch\n"
        + anchor,
        1,
    )
if "universal_training_contract_digest" not in runner.split(
    "from trade_rl.workflows.universal_training import", 1
)[-1] if "from trade_rl.workflows.universal_training import" in runner else True:
    anchor = "from trade_rl.workflows.universal_teacher_runtime import (\n"
    if anchor not in runner:
        raise SystemExit("runner teacher-runtime import anchor not found")
    runner = runner.replace(
        anchor,
        "from trade_rl.workflows.universal_training import (\n"
        "    universal_training_contract_digest,\n"
        ")\n"
        + anchor,
        1,
    )

factory_marker = "\n\ndef assemble_universal_sb3_training_backend(\n"
if "class UniversalTrainingRuntime:" not in runner:
    if factory_marker not in runner:
        raise SystemExit("runner assembly marker not found")
    block = r'''


@dataclass(frozen=True, slots=True)
class UniversalTrainingRuntime:
    """Immutable candidate-specific Universal runtime identity."""

    train_symbols: tuple[str, ...]
    catalog_digest: str
    partition_digest: str
    split_manifest_digest: str
    feature_schema_digest: str
    statistics_digest: str
    instrument_context_schema_digest: str
    training_contract_digest: str
    routed_environment_factory: UniversalRoutedEnvironmentFactory
    pretraining_artifact_digest: str | None = None

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("Universal runtime train_symbols must be non-empty and unique")
        for field_name, value in (
            ("catalog_digest", self.catalog_digest),
            ("partition_digest", self.partition_digest),
            ("split_manifest_digest", self.split_manifest_digest),
            ("feature_schema_digest", self.feature_schema_digest),
            ("statistics_digest", self.statistics_digest),
            ("instrument_context_schema_digest", self.instrument_context_schema_digest),
            ("training_contract_digest", self.training_contract_digest),
        ):
            require_sha256(value, field=f"Universal runtime {field_name}")
        if self.pretraining_artifact_digest is not None:
            require_sha256(
                self.pretraining_artifact_digest,
                field="Universal runtime pretraining_artifact_digest",
            )
        if not isinstance(
            self.routed_environment_factory, UniversalRoutedEnvironmentFactory
        ):
            raise TypeError(
                "Universal runtime routed_environment_factory must be UniversalRoutedEnvironmentFactory"
            )
        if self.routed_environment_factory.train_symbols != symbols:
            raise ValueError("Universal runtime routed symbol scope mismatch")
        if self.routed_environment_factory.partition_digest != self.partition_digest:
            raise ValueError("Universal runtime routed partition identity mismatch")
        if (
            self.routed_environment_factory.training_contract_digest
            != self.training_contract_digest
        ):
            raise ValueError("Universal runtime routed training contract mismatch")
        object.__setattr__(self, "train_symbols", symbols)

    def with_pretraining_artifact(self, artifact_digest: str) -> UniversalTrainingRuntime:
        require_sha256(
            artifact_digest,
            field="Universal runtime pretraining_artifact_digest",
        )
        return replace(self, pretraining_artifact_digest=artifact_digest)


def build_universal_training_runtime(
    *,
    train_symbols: Sequence[str],
    catalog_digest: str,
    partition_digest: str,
    split_manifest_digest: str,
    feature_schema_digest: str,
    statistics_digest: str,
    instrument_context_schema_digest: str,
    routed_environment_factory: UniversalRoutedEnvironmentFactory,
    training: Any,
) -> UniversalTrainingRuntime:
    """Rebind one routed factory to the exact architecture-specific training identity."""

    if not isinstance(routed_environment_factory, UniversalRoutedEnvironmentFactory):
        raise TypeError(
            "routed_environment_factory must be a UniversalRoutedEnvironmentFactory"
        )
    digest_payload = getattr(training, "digest_payload", None)
    if not callable(digest_payload):
        raise TypeError("Universal training config must expose digest_payload")
    training_config_digest = content_digest(digest_payload())
    training_contract_digest = universal_training_contract_digest(
        partition_digest=partition_digest,
        feature_schema_digest=feature_schema_digest,
        statistics_digest=statistics_digest,
        instrument_context_schema_digest=instrument_context_schema_digest,
        training_config_digest=training_config_digest,
    )
    bound_factory = replace(
        routed_environment_factory,
        training_contract_digest=training_contract_digest,
    )
    return UniversalTrainingRuntime(
        train_symbols=tuple(train_symbols),
        catalog_digest=catalog_digest,
        partition_digest=partition_digest,
        split_manifest_digest=split_manifest_digest,
        feature_schema_digest=feature_schema_digest,
        statistics_digest=statistics_digest,
        instrument_context_schema_digest=instrument_context_schema_digest,
        training_contract_digest=training_contract_digest,
        routed_environment_factory=bound_factory,
    )
'''
    runner = runner.replace(factory_marker, block + factory_marker, 1)

old_signature = '''def assemble_universal_sb3_training_backend(
    *,
    routed_environment_factory: UniversalRoutedEnvironmentFactory,
    training: Any,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    verbose: int = 0,
) -> tuple[StableBaselines3Backend, UniversalPretrainingBundle]:
'''
new_signature = '''def assemble_universal_sb3_training_backend(
    *,
    routed_environment_factory: UniversalRoutedEnvironmentFactory,
    training: Any,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    oracle_batches: Mapping[str, EpisodeOracleBatch] | None = None,
    verbose: int = 0,
) -> tuple[StableBaselines3Backend, UniversalPretrainingBundle]:
'''
if old_signature in runner:
    runner = runner.replace(old_signature, new_signature, 1)
elif "oracle_batches: Mapping[str, EpisodeOracleBatch] | None = None" not in runner:
    raise SystemExit("runner U4 assembly signature not found")

old_batches = '''    batches = build_universal_oracle_batches(
        train_symbols=routed_environment_factory.train_symbols,
        bindings=routed_environment_factory.bindings,
        concrete_environment_factory=(
            routed_environment_factory.concrete_environment_factory
        ),
        fold_train_range=fold_train_range,
        behavior_cloning_seed=behavior_cloning_seed,
        n_envs=n_envs,
    )
'''
new_batches = '''    if oracle_batches is None:
        batches = build_universal_oracle_batches(
            train_symbols=routed_environment_factory.train_symbols,
            bindings=routed_environment_factory.bindings,
            concrete_environment_factory=(
                routed_environment_factory.concrete_environment_factory
            ),
            fold_train_range=fold_train_range,
            behavior_cloning_seed=behavior_cloning_seed,
            n_envs=n_envs,
        )
    else:
        batches = dict(oracle_batches)
        if set(batches) != set(routed_environment_factory.train_symbols):
            raise ValueError(
                "Universal U4 oracle_batches must exactly match train_symbols"
            )
        if any(not isinstance(batch, EpisodeOracleBatch) for batch in batches.values()):
            raise TypeError("Universal U4 oracle_batches must contain EpisodeOracleBatch")
'''
if old_batches in runner:
    runner = runner.replace(old_batches, new_batches, 1)
elif "Universal U4 oracle_batches must exactly match train_symbols" not in runner:
    raise SystemExit("runner Oracle batch assembly block not found")

all_marker = "__all__ = [\n"
for exported in (
    '    "UniversalTrainingRuntime",\n',
    '    "build_universal_training_runtime",\n',
):
    if exported not in runner:
        if all_marker not in runner:
            raise SystemExit("runner __all__ marker not found")
        runner = runner.replace(all_marker, all_marker + exported, 1)

runner_path.write_text(runner)
compile(runner, str(runner_path), "exec")

stage_path = Path("trade_rl/workflows/universal_stage_a_training.py")
if not stage_path.exists():
    stage_path.write_text(r'''"""U5 training orchestration for the four Universal architecture candidates."""

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

    static_identities = {_static_runtime_identity(runtime) for _, _, runtime in prepared}
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
''')
compile(stage_path.read_text(), str(stage_path), "exec")
