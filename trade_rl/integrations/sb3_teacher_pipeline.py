"""Teacher generation, caching, and artifact lifecycle for SB3 training."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.catalog.contracts import ArtifactKind
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.integrations.sb3_behavior_cloning import (
    _teacher_cache_key,
    _TeacherIdentity,
)
from trade_rl.integrations.sb3_runtime import _oracle_accelerator_backend
from trade_rl.learning import (
    OracleTeacherConfig,
    SupervisedPolicyDataset,
    collect_teacher_rollout,
    load_teacher_artifact,
    oracle_target_path,
    write_teacher_artifact,
)
from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeSamplingConfig,
    build_episode_oracle_batch,
)
from trade_rl.learning.episode_teacher_artifact import (
    EPISODE_TEACHER_ARTIFACT_SCHEMA,
    EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
    EpisodeSupervisedPolicyDataset,
    collect_episode_teacher_rollout,
    collect_episode_teacher_rollout_parallel,
    load_episode_teacher_artifact,
    write_episode_teacher_artifact,
)
from trade_rl.learning.oracle_bellman_contracts import OracleSolverConfig
from trade_rl.learning.teacher_cache import (
    teacher_cache_identity,
    teacher_cache_identity_v2,
)


class _StableBaselines3TeacherPipeline:
    """Own teacher computation and immutable cache publication."""

    environment_factory: Callable[[], Any]
    teacher_cache_root: Path | None
    reusable_artifact_index: ReusableArtifactIndex | None
    _oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray]
    _oracle_episode_batch_cache: dict[
        tuple[str, int, int, str, str, str], EpisodeOracleBatch
    ]
    _trend_target_cache: dict[tuple[str, int, int, str], np.ndarray]
    _teacher_dataset_cache: dict[
        tuple[str, int, int, str, str, str], SupervisedPolicyDataset
    ]
    _episode_teacher_dataset_cache: dict[
        tuple[str, int, int, str, str, str],
        EpisodeSupervisedPolicyDataset,
    ]

    def _oracle_episode_batch(
        self,
        environment: Any,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
        sampling_config: OracleEpisodeSamplingConfig,
        max_workers: int = 1,
        solver_config: OracleSolverConfig | None = None,
    ) -> EpisodeOracleBatch:
        resolved_solver_config = solver_config or OracleSolverConfig()
        dataset = environment.dataset
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("Oracle episode dataset must expose dataset_id")
        start, stop = train_range
        key = (
            dataset_id,
            int(start),
            int(stop),
            teacher_config.digest,
            sampling_config.digest,
            resolved_solver_config.digest,
        )
        cached = self._oracle_episode_batch_cache.get(key)
        if cached is not None:
            return cached
        batch = build_episode_oracle_batch(
            dataset,
            minimum_start_index=start,
            sampling_config=sampling_config,
            teacher_config=teacher_config,
            initial_weight_provider=lambda mode, index: resolve_episode_initial_weights(
                environment,
                mode,
                index,
            ),
            max_workers=max_workers,
            solver_config=resolved_solver_config,
            accelerator_backend=_oracle_accelerator_backend(resolved_solver_config),
        )
        self._oracle_episode_batch_cache[key] = batch
        return batch

    def _episode_teacher_dataset(
        self,
        environment: Any,
        batch: EpisodeOracleBatch,
        *,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
        max_workers: int = 1,
    ) -> EpisodeSupervisedPolicyDataset:
        start, stop = train_range
        environment_digest = getattr(environment, "environment_digest", None)
        action_spec_digest = getattr(environment, "action_spec_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError(
                "episode teacher environment must expose environment_digest"
            )
        if not isinstance(action_spec_digest, str):
            raise ValueError(
                "episode teacher environment must expose action_spec_digest"
            )
        artifact_schema = (
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V1
            if batch.solver_provenance is None
            else EPISODE_TEACHER_ARTIFACT_SCHEMA
        )
        teacher_identity = content_digest(
            {
                "episode_batch_digest": batch.digest,
                "schema_version": artifact_schema,
                "teacher_config_digest": teacher_config.digest,
            }
        )
        key = (
            batch.dataset_id,
            int(start),
            int(stop),
            environment_digest,
            action_spec_digest,
            teacher_identity,
        )
        cached = self._episode_teacher_dataset_cache.get(key)
        if cached is not None:
            return cached
        cache_path: Path | None = None
        shard_root: Path | None = None
        if self.teacher_cache_root is not None:
            cache_identity = (
                teacher_cache_identity(
                    dataset_id=batch.dataset_id,
                    train_range=(start, stop),
                    environment_digest=environment_digest,
                    action_spec_digest=action_spec_digest,
                    teacher_config_digest=teacher_identity,
                )
                if batch.solver_provenance is None
                else teacher_cache_identity_v2(
                    dataset_id=batch.dataset_id,
                    train_range=(start, stop),
                    environment_digest=environment_digest,
                    action_spec_digest=action_spec_digest,
                    teacher_config_digest=teacher_identity,
                    solver_provenance=batch.solver_provenance,
                )
            )
            cache_path = self.teacher_cache_root / _teacher_cache_key(
                dataset_id=batch.dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_identity,
            )
            if self.reusable_artifact_index is not None:
                indexed = self.reusable_artifact_index.resolve(
                    ArtifactKind.ORACLE_TEACHER,
                    cache_identity,
                )
                if indexed is not None:
                    cache_path = indexed
            shard_root = cache_path.with_name(f".{cache_path.name}.episodes")
            if cache_path.exists():
                manifest, teacher_dataset = load_episode_teacher_artifact(
                    cache_path,
                    expected_dataset_id=batch.dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                )
                if teacher_dataset.teacher_config_digest != teacher_identity:
                    raise ValueError("cached episode teacher identity mismatch")
                if self.reusable_artifact_index is not None:
                    self.reusable_artifact_index.register_directory(
                        artifact_digest=manifest.artifact_digest,
                        artifact_kind=ArtifactKind.ORACLE_TEACHER,
                        schema_version=manifest.schema_version,
                        dataset_id=batch.dataset_id,
                        cache_key=cache_identity,
                        metadata={
                            "episode_count": manifest.episode_count,
                            "sample_count": manifest.sample_count,
                            **(
                                {}
                                if manifest.solver_provenance is None
                                else {
                                    "solver_provenance": manifest.solver_provenance.serialized_payload()
                                }
                            ),
                        },
                        location=cache_path,
                    )
                self._episode_teacher_dataset_cache[key] = teacher_dataset
                return teacher_dataset
        if max_workers == 1:
            teacher_dataset = collect_episode_teacher_rollout(
                environment,
                batch,
                teacher_config_digest=teacher_identity,
            )
        else:
            teacher_dataset = collect_episode_teacher_rollout_parallel(
                self.environment_factory,
                batch,
                teacher_config_digest=teacher_identity,
                max_workers=max_workers,
                shard_root=shard_root,
            )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_path.name}.", dir=str(cache_path.parent)
                )
            )
            try:
                write_episode_teacher_artifact(temporary, teacher_dataset)
                try:
                    temporary.replace(cache_path)
                except FileExistsError:
                    pass
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            if shard_root is not None and shard_root.exists():
                shutil.rmtree(shard_root)
            if self.reusable_artifact_index is not None:
                manifest, _ = load_episode_teacher_artifact(
                    cache_path,
                    expected_dataset_id=batch.dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                )
                self.reusable_artifact_index.register_directory(
                    artifact_digest=manifest.artifact_digest,
                    artifact_kind=ArtifactKind.ORACLE_TEACHER,
                    schema_version=manifest.schema_version,
                    dataset_id=batch.dataset_id,
                    cache_key=cache_identity,
                    metadata={
                        "episode_count": manifest.episode_count,
                        "sample_count": manifest.sample_count,
                    },
                    location=cache_path,
                )
        self._episode_teacher_dataset_cache[key] = teacher_dataset
        return teacher_dataset

    def _oracle_targets(
        self,
        dataset: Any,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
    ) -> np.ndarray:
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("oracle dataset must expose dataset_id")
        start, stop = train_range
        key = (dataset_id, int(start), int(stop), teacher_config.digest)
        cached = self._oracle_target_cache.get(key)
        if cached is not None:
            return cached
        targets = np.asarray(
            oracle_target_path(dataset, train_range, teacher_config),
            dtype=np.float32,
        ).copy(order="C")
        targets.setflags(write=False)
        self._oracle_target_cache[key] = targets
        return targets

    def _trend_baseline_targets(
        self,
        dataset: Any,
        train_range: tuple[int, int],
        strategy: Any,
        *,
        teacher_digest: str,
    ) -> np.ndarray:
        """Return causal base-trend targets aligned with policy decisions."""

        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("trend teacher dataset must expose dataset_id")
        start, stop = train_range
        key = (dataset_id, int(start), int(stop), teacher_digest)
        cached = self._trend_target_cache.get(key)
        if cached is not None:
            return cached
        targets = np.stack(
            [
                np.asarray(strategy.targets(dataset, index).base, dtype=np.float32)
                for index in range(start, stop - 1)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if targets.ndim != 2 or len(targets) != stop - start - 1:
            raise RuntimeError("causal trend teacher target shape mismatch")
        if not np.isfinite(targets).all():
            raise RuntimeError("causal trend teacher targets are non-finite")
        targets.setflags(write=False)
        self._trend_target_cache[key] = targets
        return targets

    def _teacher_dataset(
        self,
        environment: Any,
        targets: np.ndarray,
        *,
        dataset_id: str,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig | _TeacherIdentity,
    ) -> SupervisedPolicyDataset:
        start, stop = train_range
        environment_digest = getattr(environment, "environment_digest", None)
        action_spec_digest = getattr(environment, "action_spec_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError("teacher environment must expose environment_digest")
        if not isinstance(action_spec_digest, str):
            raise ValueError("teacher environment must expose action_spec_digest")
        key = (
            dataset_id,
            int(start),
            int(stop),
            environment_digest,
            action_spec_digest,
            teacher_config.digest,
        )
        cached = self._teacher_dataset_cache.get(key)
        if cached is not None:
            return cached
        cache_path: Path | None = None
        if self.teacher_cache_root is not None:
            cache_identity = teacher_cache_identity(
                dataset_id=dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config.digest,
            )
            cache_path = self.teacher_cache_root / _teacher_cache_key(
                dataset_id=dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config.digest,
            )
            if self.reusable_artifact_index is not None:
                indexed = self.reusable_artifact_index.resolve(
                    ArtifactKind.ORACLE_TEACHER,
                    cache_identity,
                )
                if indexed is not None:
                    cache_path = indexed
            if cache_path.exists():
                manifest, teacher_dataset = load_teacher_artifact(
                    cache_path,
                    expected_dataset_id=dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                    expected_train_range=(start, stop),
                )
                if teacher_dataset.teacher_config_digest != teacher_config.digest:
                    raise ValueError("cached teacher configuration identity mismatch")
                if self.reusable_artifact_index is not None:
                    self.reusable_artifact_index.register_directory(
                        artifact_digest=manifest.artifact_digest,
                        artifact_kind=ArtifactKind.ORACLE_TEACHER,
                        schema_version=manifest.schema_version,
                        dataset_id=dataset_id,
                        cache_key=cache_identity,
                        metadata={"sample_count": manifest.sample_count},
                        location=cache_path,
                    )
                self._teacher_dataset_cache[key] = teacher_dataset
                return teacher_dataset
        teacher_dataset = collect_teacher_rollout(
            environment,
            targets,
            dataset_id=dataset_id,
            train_range=(start, stop),
            teacher_config_digest=teacher_config.digest,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{cache_path.name}.", dir=str(cache_path.parent)
                )
            )
            try:
                write_teacher_artifact(temporary, teacher_dataset)
                try:
                    temporary.replace(cache_path)
                except FileExistsError:
                    # A concurrent equivalent trainer won the content-addressed race.
                    pass
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            if self.reusable_artifact_index is not None:
                manifest, _ = load_teacher_artifact(
                    cache_path,
                    expected_dataset_id=dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                    expected_train_range=(start, stop),
                )
                self.reusable_artifact_index.register_directory(
                    artifact_digest=manifest.artifact_digest,
                    artifact_kind=ArtifactKind.ORACLE_TEACHER,
                    schema_version=manifest.schema_version,
                    dataset_id=dataset_id,
                    cache_key=cache_identity,
                    metadata={"sample_count": manifest.sample_count},
                    location=cache_path,
                )
        self._teacher_dataset_cache[key] = teacher_dataset
        return teacher_dataset
