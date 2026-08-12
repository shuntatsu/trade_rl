"""U6 training orchestration for the selected Universal architecture."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)
from trade_rl.workflows.universal_causal_alpha_teacher import (
    build_universal_causal_alpha_teacher_package,
)
from trade_rl.workflows.universal_research import (
    FullResearchAlgorithm,
    build_full_research_pair_closure,
)
from trade_rl.workflows.universal_teacher_runtime import build_universal_teacher_batches
from trade_rl.workflows.universal_training_runner import (
    UniversalTrainingRuntime,
    assemble_universal_sb3_training_backend,
    train_universal_seeds,
)

UniversalFullResearchRuntimeFactory = Callable[
    [FullResearchAlgorithm, ResidualTrainingConfig], UniversalTrainingRuntime
]


def _training_payload(config: ResidualTrainingConfig) -> dict[str, object]:
    return dict(config.digest_payload())


def _strip_algorithm_family_fields(
    config: ResidualTrainingConfig,
    *,
    remove_gamma: bool,
) -> dict[str, object]:
    payload = _training_payload(config)
    payload.pop("algorithm", None)
    payload.pop("cost_critic", None)
    payload.pop("lagrangian", None)
    if remove_gamma:
        payload.pop("gamma", None)
        payload.pop("discount_half_life_hours", None)
    for key in tuple(payload):
        if key.startswith("cost_") or key.startswith("lagrangian_"):
            payload.pop(key)
    return payload


def _strip_gamma(config: ResidualTrainingConfig) -> dict[str, object]:
    payload = _training_payload(config)
    payload.pop("gamma", None)
    payload.pop("discount_half_life_hours", None)
    return payload


@dataclass(frozen=True, slots=True)
class UniversalFullResearchTrainingSpec:
    """One selected-architecture training configuration in the U6 comparison."""

    algorithm: FullResearchAlgorithm
    selected_architecture: UniversalArchitectureName
    training_config: ResidualTrainingConfig

    @property
    def fixed_condition_digest(self) -> str:
        return content_digest(
            {
                "schema_version": "universal_full_research_fixed_conditions_v1",
                "training": _strip_algorithm_family_fields(
                    self.training_config,
                    remove_gamma=True,
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class UniversalFullResearchAlgorithmRun:
    """One completed algorithm-training run before paired research evaluation."""

    algorithm: FullResearchAlgorithm
    selected_architecture: UniversalArchitectureName
    training_config: ResidualTrainingConfig
    training_manifest: Mapping[str, object]
    output_root: Path

    def __post_init__(self) -> None:
        manifest = dict(self.training_manifest)
        if manifest.get("schema_version") != "universal_training_run_v1":
            raise ValueError("U6 training manifest schema mismatch")
        if manifest.get("architecture_name") != self.selected_architecture.value:
            raise ValueError("U6 training manifest architecture mismatch")
        expected_config_digest = content_digest(self.training_config.digest_payload())
        if manifest.get("training_config_digest") != expected_config_digest:
            raise ValueError("U6 training manifest config digest mismatch")
        run_digest = manifest.get("run_digest")
        if not isinstance(run_digest, str):
            raise ValueError("U6 training manifest run digest is unavailable")
        require_sha256(run_digest, field="U6 training run_digest")
        payload = {key: value for key, value in manifest.items() if key != "run_digest"}
        if content_digest(payload) != run_digest:
            raise ValueError("U6 training manifest run digest mismatch")
        object.__setattr__(self, "training_manifest", manifest)
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def run_digest(self) -> str:
        value = self.training_manifest["run_digest"]
        if not isinstance(value, str):  # pragma: no cover - validated in __post_init__
            raise RuntimeError("U6 training run digest disappeared")
        return value

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "algorithm": self.algorithm.value,
                "run_digest": self.run_digest,
                "schema_version": "universal_full_research_algorithm_run_v1",
                "selected_architecture": self.selected_architecture.value,
                "training_config_digest": content_digest(
                    self.training_config.digest_payload()
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class UniversalFullResearchTrainingComparison:
    """Training-side U6 closure; paired economic evidence remains intentionally empty."""

    selected_architecture: UniversalArchitectureName
    runs: tuple[UniversalFullResearchAlgorithmRun, ...]
    required_pairs: tuple[str, ...]
    completed_pairs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if tuple(run.algorithm for run in runs) != tuple(FullResearchAlgorithm):
            raise ValueError("U6 training comparison requires exact algorithm closure")
        if any(
            run.selected_architecture is not self.selected_architecture for run in runs
        ):
            raise ValueError("U6 training comparison architecture identity mismatch")
        required = tuple(self.required_pairs)
        completed = tuple(self.completed_pairs)
        if not required or len(set(required)) != len(required):
            raise ValueError("U6 required pair closure must be non-empty and unique")
        if len(set(completed)) != len(completed):
            raise ValueError("U6 completed pair closure must be unique")
        if not set(completed).issubset(required):
            raise ValueError(
                "U6 completed pairs contain values outside the required closure"
            )
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "required_pairs", required)
        object.__setattr__(self, "completed_pairs", completed)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "completed_pairs": list(self.completed_pairs),
                "required_pairs": list(self.required_pairs),
                "run_digests": [run.digest for run in self.runs],
                "schema_version": "universal_full_research_training_comparison_v1",
                "selected_architecture": self.selected_architecture.value,
            }
        )

    def with_completed_pairs(
        self,
        completed_pairs: Sequence[str],
    ) -> UniversalFullResearchTrainingComparison:
        return replace(self, completed_pairs=tuple(completed_pairs))


def prepare_universal_full_research_training_configs(
    *,
    selected_architecture: UniversalArchitectureName | str,
    algorithm_configs: Mapping[FullResearchAlgorithm | str, ResidualTrainingConfig],
) -> tuple[UniversalFullResearchTrainingSpec, ...]:
    """Project the selected U5 architecture onto the exact maintained U6 algorithms."""

    architecture = UniversalArchitectureName(selected_architecture)
    resolved: dict[FullResearchAlgorithm, ResidualTrainingConfig] = {}
    for raw_algorithm, config in algorithm_configs.items():
        algorithm = FullResearchAlgorithm(raw_algorithm)
        if algorithm in resolved:
            raise ValueError("U6 algorithm configuration contains duplicate algorithms")
        if not isinstance(config, ResidualTrainingConfig):
            raise TypeError(
                "U6 algorithm configurations must be ResidualTrainingConfig"
            )
        resolved[algorithm] = apply_architecture_to_training_config(
            config, architecture
        )
    required_algorithms = tuple(FullResearchAlgorithm)
    if set(resolved) != set(required_algorithms):
        raise ValueError(
            "U6 algorithm configurations must close PPO/Lagrangian/discounted"
        )

    ppo = resolved[FullResearchAlgorithm.PPO]
    lagrangian = resolved[FullResearchAlgorithm.LAGRANGIAN]
    discounted = resolved[FullResearchAlgorithm.DISCOUNTED]
    if ppo.algorithm != "ppo" or ppo.gamma != 1.0:
        raise ValueError("U6 PPO control requires algorithm=ppo and gamma=1")
    if lagrangian.algorithm != "lagrangian_ppo" or lagrangian.gamma != 1.0:
        raise ValueError(
            "U6 Lagrangian PPO requires algorithm=lagrangian_ppo and gamma=1"
        )
    if discounted.algorithm != "lagrangian_ppo" or not 0.0 < discounted.gamma < 1.0:
        raise ValueError(
            "U6 discounted comparison requires Discounted Lagrangian PPO with gamma in (0, 1)"
        )

    if _strip_algorithm_family_fields(ppo, remove_gamma=False) != (
        _strip_algorithm_family_fields(lagrangian, remove_gamma=False)
    ):
        raise ValueError(
            "U6 PPO and Lagrangian non-algorithm training conditions must match"
        )
    if _strip_gamma(lagrangian) != _strip_gamma(discounted):
        raise ValueError(
            "U6 Lagrangian and discounted configurations may differ only by gamma"
        )
    if tuple(ppo.seeds) != tuple(lagrangian.seeds) or tuple(ppo.seeds) != tuple(
        discounted.seeds
    ):
        raise ValueError("U6 algorithms must use identical seed closure")

    specs = tuple(
        UniversalFullResearchTrainingSpec(
            algorithm=algorithm,
            selected_architecture=architecture,
            training_config=resolved[algorithm],
        )
        for algorithm in required_algorithms
    )
    if len({spec.fixed_condition_digest for spec in specs}) != 1:
        raise ValueError(
            "U6 algorithms contain unexpected non-comparison condition drift"
        )
    return specs


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


def train_universal_full_research_comparison(
    *,
    selected_architecture: UniversalArchitectureName | str,
    algorithm_configs: Mapping[FullResearchAlgorithm | str, ResidualTrainingConfig],
    runtime_factory: UniversalFullResearchRuntimeFactory,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    baseline_names: Sequence[str],
    folds: Sequence[int],
    output_root: Path,
    verbose: int = 0,
) -> UniversalFullResearchTrainingComparison:
    """Train the selected architecture across the maintained three-algorithm U6 screen."""

    architecture = UniversalArchitectureName(selected_architecture)
    if not callable(runtime_factory):
        raise TypeError("U6 runtime_factory must be callable")
    require_sha256(normalizer_digest, field="U6 normalizer_digest")
    require_sha256(feature_schema_digest, field="U6 feature_schema_digest")
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
        raise ValueError("U6 verbose must be a non-negative integer")
    specs = prepare_universal_full_research_training_configs(
        selected_architecture=architecture,
        algorithm_configs=algorithm_configs,
    )

    prepared: list[
        tuple[UniversalFullResearchTrainingSpec, UniversalTrainingRuntime]
    ] = []
    for spec in specs:
        runtime = runtime_factory(spec.algorithm, spec.training_config)
        if not isinstance(runtime, UniversalTrainingRuntime):
            raise TypeError("U6 runtime_factory must return UniversalTrainingRuntime")
        if runtime.feature_schema_digest != feature_schema_digest:
            raise ValueError("U6 runtime feature schema identity mismatch")
        prepared.append((spec, runtime))
    if len({_static_runtime_identity(runtime) for _, runtime in prepared}) != 1:
        raise ValueError("U6 non-algorithm runtime conditions must be identical")

    first_spec, first_runtime = prepared[0]
    behavior_cloning_seed = first_spec.training_config.behavior_cloning_seed
    if (
        isinstance(behavior_cloning_seed, bool)
        or not isinstance(behavior_cloning_seed, int)
        or not 0 <= behavior_cloning_seed <= 0xFFFFFFFF
    ):
        raise ValueError("U6 requires an explicit uint32 behavior_cloning_seed")

    teacher_kind = first_spec.training_config.behavior_cloning_teacher
    if any(
        spec.training_config.behavior_cloning_teacher != teacher_kind
        for spec, _ in prepared
    ):
        raise ValueError("U6 algorithms must share one BC teacher kind")
    if teacher_kind == "causal_alpha_ridge":
        routed = first_runtime.routed_environment_factory
        shared_causal_package = build_universal_causal_alpha_teacher_package(
            train_symbols=first_runtime.train_symbols,
            bindings=routed.bindings,
            concrete_environment_factory=routed.concrete_environment_factory,
            instrument_context_provider=routed.instrument_context_provider,
            fold_train_range=fold_train_range,
            feature_schema_digest=feature_schema_digest,
        )
        shared_batches = None
    else:
        shared_causal_package = None
        shared_batches = build_universal_teacher_batches(
            teacher_kind=teacher_kind,
            train_symbols=first_runtime.train_symbols,
            bindings=first_runtime.routed_environment_factory.bindings,
            concrete_environment_factory=(
                first_runtime.routed_environment_factory.concrete_environment_factory
            ),
            fold_train_range=fold_train_range,
            behavior_cloning_seed=behavior_cloning_seed,
            n_envs=first_spec.training_config.n_envs,
        )

    runs: list[UniversalFullResearchAlgorithmRun] = []
    root = Path(output_root)
    for spec, runtime in prepared:
        backend, bundle = assemble_universal_sb3_training_backend(
            routed_environment_factory=runtime.routed_environment_factory,
            training=spec.training_config,
            fold_train_range=fold_train_range,
            normalizer_digest=normalizer_digest,
            feature_schema_digest=feature_schema_digest,
            oracle_batches=shared_batches,
            causal_teacher_package=shared_causal_package,
            verbose=verbose,
        )
        bound_runtime = runtime.with_pretraining_artifact(
            bundle.teacher_artifact.artifact_digest
        )
        algorithm_root = root / spec.algorithm.value
        manifest = train_universal_seeds(
            runtime=bound_runtime,
            training=spec.training_config,
            backend=backend,
            output_root=algorithm_root,
            architecture_name=architecture.value,
        )
        runs.append(
            UniversalFullResearchAlgorithmRun(
                algorithm=spec.algorithm,
                selected_architecture=architecture,
                training_config=spec.training_config,
                training_manifest=manifest,
                output_root=algorithm_root,
            )
        )

    required_pairs = build_full_research_pair_closure(
        algorithms=tuple(FullResearchAlgorithm),
        baseline_names=tuple(baseline_names),
        folds=tuple(folds),
        seeds=tuple(specs[0].training_config.seeds),
    )
    return UniversalFullResearchTrainingComparison(
        selected_architecture=architecture,
        runs=tuple(runs),
        required_pairs=required_pairs,
        completed_pairs=(),
    )


__all__ = [
    "UniversalFullResearchAlgorithmRun",
    "UniversalFullResearchRuntimeFactory",
    "UniversalFullResearchTrainingComparison",
    "UniversalFullResearchTrainingSpec",
    "prepare_universal_full_research_training_configs",
    "train_universal_full_research_comparison",
]
