"""Strict entrypoint assembly for executable Universal U6 training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.runtime_factory import (
    RuntimeFactoryDescriptor,
    load_runtime_factory,
)
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
    load_causal_alpha_v4_context_manifest,
    validate_causal_alpha_v4_context_manifest_against_base,
)
from trade_rl.workflows.universal_full_research_training import (
    UniversalFullResearchTrainingComparison,
    train_universal_full_research_comparison,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_runtime_manifest import (
    UniversalRuntimeManifest,
    load_universal_runtime_manifest,
)
from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime

UniversalEntrypointRuntimeFactory = Callable[..., UniversalTrainingRuntime]


@dataclass(frozen=True, slots=True)
class UniversalRuntimeFactoryContext:
    """Explicit external inputs available to a real Universal runtime factory."""

    runtime_manifest_path: Path
    frozen_metadata_root: Path
    instrument_artifact_root: Path | None = None
    dataset_artifact_root: Path | None = None
    fold_train_range: tuple[int, int] | None = None
    normalizer_digest: str | None = None
    feature_schema_digest: str | None = None
    v4_context_manifest_path: Path | None = None
    manifest: UniversalRuntimeManifest = field(init=False)
    normalizer_artifact_root: Path = field(init=False)
    v4_context_manifest: CausalAlphaV4ContextManifest | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        manifest_path = Path(self.runtime_manifest_path)
        frozen_root = Path(self.frozen_metadata_root)
        manifest = load_universal_runtime_manifest(manifest_path)
        v4_manifest_path = (
            None
            if self.v4_context_manifest_path is None
            else Path(self.v4_context_manifest_path)
        )
        v4_manifest: CausalAlphaV4ContextManifest | None = None
        if v4_manifest_path is not None:
            v4_manifest = load_causal_alpha_v4_context_manifest(v4_manifest_path)
            validate_causal_alpha_v4_context_manifest_against_base(
                v4_manifest, manifest
            )
        base = manifest_path.parent
        instrument_root = base / manifest.instrument_artifact_relpath
        dataset_root = base / manifest.dataset_artifact_relpath
        normalizer_root = base / manifest.normalizer_artifact_relpath
        if (
            self.instrument_artifact_root is not None
            and Path(self.instrument_artifact_root).resolve()
            != instrument_root.resolve()
        ):
            raise ValueError("instrument artifact root compatibility mismatch")
        if (
            self.dataset_artifact_root is not None
            and Path(self.dataset_artifact_root).resolve() != dataset_root.resolve()
        ):
            raise ValueError("dataset artifact root compatibility mismatch")
        if (
            self.fold_train_range is not None
            and self.fold_train_range != manifest.fold_train_range
        ):
            raise ValueError("fold train range compatibility mismatch")
        if self.normalizer_digest is not None:
            require_sha256(self.normalizer_digest, field="normalizer_digest")
            if self.normalizer_digest != manifest.statistics_digest:
                raise ValueError("normalizer digest compatibility mismatch")
        if self.feature_schema_digest is not None:
            require_sha256(self.feature_schema_digest, field="feature_schema_digest")
            if self.feature_schema_digest != manifest.feature_schema_digest:
                raise ValueError("feature schema digest compatibility mismatch")
        object.__setattr__(self, "runtime_manifest_path", manifest_path)
        object.__setattr__(self, "frozen_metadata_root", frozen_root)
        object.__setattr__(self, "instrument_artifact_root", instrument_root)
        object.__setattr__(self, "dataset_artifact_root", dataset_root)
        object.__setattr__(self, "normalizer_artifact_root", normalizer_root)
        object.__setattr__(self, "v4_context_manifest_path", v4_manifest_path)
        object.__setattr__(self, "v4_context_manifest", v4_manifest)
        object.__setattr__(self, "fold_train_range", manifest.fold_train_range)
        object.__setattr__(self, "normalizer_digest", manifest.statistics_digest)
        object.__setattr__(
            self, "feature_schema_digest", manifest.feature_schema_digest
        )
        object.__setattr__(self, "manifest", manifest)

    @property
    def resolved_instrument_artifact_root(self) -> Path:
        value = self.instrument_artifact_root
        if value is None:  # pragma: no cover - established in __post_init__
            raise RuntimeError("instrument artifact root was not resolved")
        return value

    @property
    def resolved_dataset_artifact_root(self) -> Path:
        value = self.dataset_artifact_root
        if value is None:  # pragma: no cover - established in __post_init__
            raise RuntimeError("dataset artifact root was not resolved")
        return value


@dataclass(frozen=True, slots=True)
class UniversalFullResearchEntrypointResult:
    manifest_path: Path
    manifest_digest: str
    comparison_digest: str
    research_success: bool = False

    def __post_init__(self) -> None:
        require_sha256(self.manifest_digest, field="entrypoint manifest_digest")
        require_sha256(self.comparison_digest, field="entrypoint comparison_digest")
        if self.research_success:
            raise ValueError(
                "training entrypoint cannot claim research success before paired sealed evidence"
            )
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))


def load_universal_runtime_factory(spec: str) -> UniversalEntrypointRuntimeFactory:
    """Load one explicit `module:function` Universal runtime factory."""

    return cast(UniversalEntrypointRuntimeFactory, load_runtime_factory(spec))


def _resolved_run_configs(
    run_configs: Mapping[FullResearchAlgorithm | str, TrainingRunConfig],
) -> dict[FullResearchAlgorithm, TrainingRunConfig]:
    resolved: dict[FullResearchAlgorithm, TrainingRunConfig] = {}
    for raw_algorithm, config in run_configs.items():
        algorithm = FullResearchAlgorithm(raw_algorithm)
        if algorithm in resolved:
            raise ValueError("Universal U6 run configs contain duplicate algorithms")
        if not isinstance(config, TrainingRunConfig):
            raise TypeError("Universal U6 run configs must be TrainingRunConfig")
        resolved[algorithm] = config
    if set(resolved) != set(FullResearchAlgorithm):
        raise ValueError(
            "Universal U6 run configs must close all maintained algorithms"
        )
    return resolved


def _non_training_identity(config: TrainingRunConfig) -> str:
    payload = dict(config.candidate_digest_payload())
    payload.pop("training", None)
    return content_digest(payload)


def run_universal_full_research_training(
    *,
    selected_architecture: UniversalArchitectureName | str,
    run_configs: Mapping[FullResearchAlgorithm | str, TrainingRunConfig],
    runtime_factory: UniversalEntrypointRuntimeFactory,
    runtime_factory_descriptor: RuntimeFactoryDescriptor | None = None,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    baseline_names: Sequence[str],
    folds: Sequence[int],
    output_root: Path,
    verbose: int = 0,
) -> UniversalFullResearchEntrypointResult:
    """Execute U6 training while keeping authored non-training surfaces identical."""

    architecture = UniversalArchitectureName(selected_architecture)
    configs = _resolved_run_configs(run_configs)
    if not callable(runtime_factory):
        raise TypeError("runtime_factory must be callable")
    if runtime_factory_descriptor is not None and not isinstance(
        runtime_factory_descriptor, RuntimeFactoryDescriptor
    ):
        raise TypeError(
            "runtime_factory_descriptor must be RuntimeFactoryDescriptor or null"
        )
    if len({_non_training_identity(config) for config in configs.values()}) != 1:
        raise ValueError("Universal U6 non-training run surfaces must be identical")
    require_sha256(normalizer_digest, field="Universal U6 normalizer_digest")
    require_sha256(feature_schema_digest, field="Universal U6 feature_schema_digest")
    baseline_values = tuple(str(value) for value in baseline_names)
    fold_values = tuple(folds)
    if not baseline_values or any(not value for value in baseline_values):
        raise ValueError("Universal U6 baseline_names must be non-empty")
    if not fold_values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in fold_values
    ):
        raise ValueError("Universal U6 folds must be non-negative integers")

    algorithm_configs: Mapping[FullResearchAlgorithm | str, ResidualTrainingConfig] = {
        algorithm: config.training for algorithm, config in configs.items()
    }

    def projected_runtime_factory(
        algorithm: FullResearchAlgorithm,
        training: ResidualTrainingConfig,
    ) -> UniversalTrainingRuntime:
        authored = configs[algorithm]
        projected = replace(authored, training=training)
        runtime = runtime_factory(algorithm=algorithm, run_config=projected)
        if not isinstance(runtime, UniversalTrainingRuntime):
            raise TypeError("runtime factory must return UniversalTrainingRuntime")
        return runtime

    comparison: UniversalFullResearchTrainingComparison = (
        train_universal_full_research_comparison(
            selected_architecture=architecture,
            algorithm_configs=algorithm_configs,
            runtime_factory=projected_runtime_factory,
            fold_train_range=fold_train_range,
            normalizer_digest=normalizer_digest,
            feature_schema_digest=feature_schema_digest,
            baseline_names=baseline_values,
            folds=fold_values,
            output_root=Path(output_root),
            verbose=verbose,
        )
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "algorithm_run_digests": {
            run.algorithm.value: run.run_digest for run in comparison.runs
        },
        "comparison_digest": comparison.digest,
        "completed_pairs": list(comparison.completed_pairs),
        "research_success": False,
        "research_success_reason": (
            "paired evaluation and sealed zero-shot evidence have not been completed"
        ),
        "required_pairs": list(comparison.required_pairs),
        "schema_version": "universal_full_research_training_entrypoint_v1",
        "selected_architecture": architecture.value,
    }
    if runtime_factory_descriptor is not None:
        payload["runtime_factory"] = runtime_factory_descriptor.to_payload()
        payload["schema_version"] = "universal_full_research_training_entrypoint_v2"
    manifest_digest = content_digest(payload)
    manifest_path = output / "universal-full-research-training.json"
    atomic_write_bytes(
        manifest_path,
        canonical_json_bytes({**payload, "manifest_digest": manifest_digest}),
    )
    return UniversalFullResearchEntrypointResult(
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        comparison_digest=comparison.digest,
        research_success=False,
    )


__all__ = [
    "UniversalEntrypointRuntimeFactory",
    "UniversalFullResearchEntrypointResult",
    "UniversalRuntimeFactoryContext",
    "load_universal_runtime_factory",
    "run_universal_full_research_training",
]
