from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trade_rl.catalog.contracts import (
    ArtifactKind,
    ArtifactRecord,
    ArtifactRegistration,
)
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex


class _Catalog:
    def __init__(self) -> None:
        self.records: dict[tuple[ArtifactKind, str], ArtifactRecord] = {}
        self.registered: ArtifactRegistration | None = None

    def find(
        self, artifact_kind: ArtifactKind, cache_key: object
    ) -> ArtifactRecord | None:
        assert isinstance(cache_key, dict)
        return self.records.get((artifact_kind, str(cache_key["identity"])))

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        self.registered = registration
        now = datetime.now(UTC)
        record = ArtifactRecord(
            registration=registration,
            created_at=now,
            last_seen_at=now,
        )
        self.records[
            (registration.artifact_kind, str(registration.cache_key["identity"]))
        ] = record
        return record


def _record(path: Path) -> ArtifactRecord:
    registration = ArtifactRegistration(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-1"},
        metadata={"sample_count": 10},
        location=str(path),
        size_bytes=4,
    )
    now = datetime.now(UTC)
    return ArtifactRecord(registration=registration, created_at=now, last_seen_at=now)


def test_resolves_ready_directory_and_registers_file_backed_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    artifact = root / "teacher"
    artifact.mkdir(parents=True)
    (artifact / "arrays.npz").write_bytes(b"data")
    catalog = _Catalog()
    catalog.records[(ArtifactKind.ORACLE_TEACHER, "teacher-1")] = _record(artifact)
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    assert (
        index.resolve(ArtifactKind.ORACLE_TEACHER, {"identity": "teacher-1"})
        == artifact
    )
    index.register_directory(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-1"},
        metadata={"sample_count": 10},
        location=artifact,
    )

    assert catalog.registered is not None
    assert catalog.registered.size_bytes == 4


def test_new_registration_namespaces_payload_digest_by_cache_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    artifact = root / "teacher"
    artifact.mkdir(parents=True)
    (artifact / "arrays.npz").write_bytes(b"data")
    catalog = _Catalog()
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    index.register_directory(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-2"},
        metadata={"sample_count": 10},
        location=artifact,
    )

    assert catalog.registered is not None
    assert catalog.registered.artifact_digest != "a" * 64
    assert catalog.registered.metadata["payload_digest"] == "a" * 64


def test_refreshes_legacy_registration_without_mutating_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    artifact = root / "teacher"
    artifact.mkdir(parents=True)
    (artifact / "arrays.npz").write_bytes(b"data")
    catalog = _Catalog()
    legacy = _record(artifact)
    catalog.records[(ArtifactKind.ORACLE_TEACHER, "teacher-1")] = legacy
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    index.register_directory(
        artifact_digest="a" * 64,
        artifact_kind=ArtifactKind.ORACLE_TEACHER,
        schema_version="episode_supervised_teacher_artifact_v1",
        dataset_id="b" * 64,
        cache_key={"identity": "teacher-1"},
        metadata={"sample_count": 10, "new_runtime_field": True},
        location=artifact,
    )

    assert catalog.registered == legacy.registration


def test_rejects_catalog_location_outside_durable_root(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    catalog = _Catalog()
    catalog.records[(ArtifactKind.ORACLE_TEACHER, "teacher-1")] = _record(outside)
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="escapes storage root"):
        index.resolve(ArtifactKind.ORACLE_TEACHER, {"identity": "teacher-1"})


def test_cache_identity_v2_shares_numerically_equivalent_backends() -> None:
    from trade_rl.learning.oracle_bellman_contracts import (
        OracleSolverConfig,
        OracleSolverProvenance,
    )
    from trade_rl.learning.teacher_cache import teacher_cache_identity_v2

    config = OracleSolverConfig(selection="numpy")
    numpy_provenance = OracleSolverProvenance.numpy_reference(
        config=config,
        market_tape_digest="c" * 64,
    )
    cuda_provenance = OracleSolverProvenance(
        backend="torch_cuda",
        solver_config_digest=config.digest,
        market_tape_digest="c" * 64,
        numeric_dtype="float64",
        tie_tolerance=config.tie_tolerance,
        episode_batch_size=config.episode_batch_size,
        target_state_block_size=config.target_state_block_size,
        compile_mode=config.compile_mode,
        compile_chunk_size=config.compile_chunk_size,
    )
    base = {
        "dataset_id": "a" * 64,
        "train_range": (1, 9),
        "environment_digest": "b" * 64,
        "action_spec_digest": "d" * 64,
        "teacher_config_digest": "e" * 64,
    }

    numpy_key = teacher_cache_identity_v2(**base, solver_provenance=numpy_provenance)
    cuda_key = teacher_cache_identity_v2(**base, solver_provenance=cuda_provenance)

    assert numpy_key == cuda_key
    assert numpy_key["schema_version"] == "teacher_cache_identity_v2"
    assert "solver_backend" not in numpy_key


class _CanonicalCatalog:
    def __init__(self) -> None:
        self.records: dict[tuple[ArtifactKind, str], ArtifactRecord] = {}
        self.registrations: list[ArtifactRegistration] = []

    @staticmethod
    def _key(
        artifact_kind: ArtifactKind, cache_key: object
    ) -> tuple[ArtifactKind, str]:
        from collections.abc import Mapping

        from trade_rl.artifacts.hashing import content_digest

        assert isinstance(cache_key, Mapping)
        return artifact_kind, content_digest(cache_key)

    def find(
        self, artifact_kind: ArtifactKind, cache_key: object
    ) -> ArtifactRecord | None:
        return self.records.get(self._key(artifact_kind, cache_key))

    def register(self, registration: ArtifactRegistration) -> ArtifactRecord:
        self.registrations.append(registration)
        now = datetime.now(UTC)
        record = ArtifactRecord(
            registration=registration,
            created_at=now,
            last_seen_at=now,
        )
        self.records[self._key(registration.artifact_kind, registration.cache_key)] = (
            record
        )
        return record


def _episode_dataset_with_provenance(*, with_provenance: bool):
    import numpy as np

    from trade_rl.learning.episode_teacher_artifact import (
        EpisodeSupervisedPolicyDataset,
    )
    from trade_rl.learning.oracle_bellman_contracts import (
        OracleSolverConfig,
        OracleSolverProvenance,
    )

    provenance = (
        OracleSolverProvenance.numpy_reference(
            config=OracleSolverConfig(),
            market_tape_digest="f" * 64,
        )
        if with_provenance
        else None
    )
    return EpisodeSupervisedPolicyDataset(
        observations=np.zeros((2, 1), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        dataset_id="a" * 64,
        train_start=1,
        train_stop=4,
        environment_digest="b" * 64,
        action_spec_digest="c" * 64,
        teacher_config_digest="d" * 64,
        decision_indices=np.array([1, 2], dtype=np.int64),
        episode_ids=np.array([0, 0], dtype=np.int64),
        solver_provenance=provenance,
    )


def test_v1_backfill_remains_legacy_without_cuda_claim(tmp_path: Path) -> None:
    from trade_rl.learning.episode_teacher_artifact import (
        write_episode_teacher_artifact,
    )
    from trade_rl.learning.teacher_cache import backfill_teacher_cache

    root = tmp_path / "cache"
    artifact = root / "legacy"
    artifact.mkdir(parents=True)
    write_episode_teacher_artifact(
        artifact,
        _episode_dataset_with_provenance(with_provenance=False),
    )
    catalog = _CanonicalCatalog()
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    assert backfill_teacher_cache(index) == 1

    registration = catalog.registrations[-1]
    assert registration.cache_key["schema_version"] == "teacher_cache_identity_v1"
    assert "solver_backend" not in registration.cache_key
    assert "solver_provenance" not in registration.metadata


def test_v2_backfill_records_runtime_evidence_outside_cache_identity(
    tmp_path: Path,
) -> None:
    from trade_rl.learning.episode_teacher_artifact import (
        write_episode_teacher_artifact,
    )
    from trade_rl.learning.teacher_cache import backfill_teacher_cache

    root = tmp_path / "cache"
    artifact = root / "current"
    artifact.mkdir(parents=True)
    write_episode_teacher_artifact(
        artifact,
        _episode_dataset_with_provenance(with_provenance=True),
    )
    catalog = _CanonicalCatalog()
    index = ReusableArtifactIndex(catalog, storage_root=root)  # type: ignore[arg-type]

    assert backfill_teacher_cache(index) == 1

    registration = catalog.registrations[-1]
    assert registration.cache_key["schema_version"] == "teacher_cache_identity_v2"
    assert "solver_backend" not in registration.cache_key
    assert registration.metadata["solver_provenance"]["backend"] == "numpy"


def test_v2_identity_ignores_runtime_metrics_and_backend() -> None:
    from dataclasses import replace

    from trade_rl.learning.oracle_bellman_contracts import (
        OracleSolverConfig,
        OracleSolverProvenance,
    )
    from trade_rl.learning.teacher_cache import teacher_cache_identity_v2

    base_provenance = OracleSolverProvenance.numpy_reference(
        config=OracleSolverConfig(),
        market_tape_digest="f" * 64,
    )
    slower = replace(
        base_provenance,
        solver_wall_time_seconds=99.0,
        peak_host_memory_bytes=999,
        digest="",
    )
    cuda = replace(base_provenance, backend="torch_cuda", digest="")
    base = {
        "dataset_id": "a" * 64,
        "train_range": (1, 4),
        "environment_digest": "b" * 64,
        "action_spec_digest": "c" * 64,
        "teacher_config_digest": "d" * 64,
    }

    normal_key = teacher_cache_identity_v2(**base, solver_provenance=base_provenance)
    slower_key = teacher_cache_identity_v2(**base, solver_provenance=slower)
    cuda_key = teacher_cache_identity_v2(**base, solver_provenance=cuda)

    assert slower_key == normal_key
    assert cuda_key == normal_key


def test_v2_artifact_digest_ignores_runtime_evidence(tmp_path: Path) -> None:
    from dataclasses import replace

    from trade_rl.learning.episode_teacher_artifact import (
        EpisodeSupervisedPolicyDataset,
        load_episode_teacher_artifact,
        write_episode_teacher_artifact,
    )

    base_dataset = _episode_dataset_with_provenance(with_provenance=True)
    assert base_dataset.solver_provenance is not None
    runtime_provenance = replace(
        base_dataset.solver_provenance,
        backend="torch_cuda",
        solver_wall_time_seconds=12.0,
        peak_device_memory_bytes=4096,
        device_name="GPU",
        digest="",
    )
    runtime_dataset = EpisodeSupervisedPolicyDataset(
        observations=base_dataset.observations,
        actions=base_dataset.actions,
        dataset_id=base_dataset.dataset_id,
        train_start=base_dataset.train_start,
        train_stop=base_dataset.train_stop,
        environment_digest=base_dataset.environment_digest,
        action_spec_digest=base_dataset.action_spec_digest,
        teacher_config_digest=base_dataset.teacher_config_digest,
        decision_indices=base_dataset.decision_indices,
        episode_ids=base_dataset.episode_ids,
        solver_provenance=runtime_provenance,
    )

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_digest = write_episode_teacher_artifact(first_root, base_dataset)
    second_digest = write_episode_teacher_artifact(second_root, runtime_dataset)
    second_manifest, _ = load_episode_teacher_artifact(second_root)

    assert second_digest == first_digest
    assert second_manifest.solver_provenance == runtime_provenance
