"""Strict entrypoint assembly for executable Universal U6 training."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.rl.universal_architecture import UniversalArchitectureName
from trade_rl.workflows.universal_full_research_training import (
    UniversalFullResearchTrainingComparison,
    train_universal_full_research_comparison,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_training_runner import UniversalTrainingRuntime

UniversalEntrypointRuntimeFactory = Callable[..., UniversalTrainingRuntime]


@dataclass(frozen=True, slots=True)
class UniversalRuntimeFactoryContext:
    """Explicit external inputs available to a real Universal runtime factory."""

    instrument_artifact_root: Path
    postgres_url: str
    dataset_artifact_root: Path
    fold_train_range: tuple[int, int]
    normalizer_digest: str
    feature_schema_digest: str

    def __post_init__(self) -> None:
        instrument_root = Path(self.instrument_artifact_root)
        dataset_root = Path(self.dataset_artifact_root)
        if not str(instrument_root):
            raise ValueError("instrument_artifact_root must not be empty")
        if not isinstance(self.postgres_url, str) or not self.postgres_url.strip():
            raise ValueError("postgres_url must be non-empty")
        if not str(dataset_root):
            raise ValueError("dataset_artifact_root must not be empty")
        start, stop = self.fold_train_range
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
        ):
            raise ValueError("fold_train_range is invalid")
        require_sha256(self.normalizer_digest, field="normalizer_digest")
        require_sha256(self.feature_schema_digest, field="feature_schema_digest")
        object.__setattr__(self, "instrument_artifact_root", instrument_root)
        object.__setattr__(self, "dataset_artifact_root", dataset_root)


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

    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("runtime factory must use module:function syntax")
    module_name, function_name = (part.strip() for part in spec.split(":", 1))
    if not module_name or not function_name:
        raise ValueError("runtime factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError("runtime factory target must be callable")
    return factory


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
