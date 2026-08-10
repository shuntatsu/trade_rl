from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from trade_rl.studio.contracts import JobSummary
from trade_rl.studio.errors import ArtifactInvalid, InvalidStudioRequest
from trade_rl.studio.training_metrics import StudioTrainingMetricsReader

from .test_catalog import settings


def _job(tmp_path: Path, *, run_id: str = "run-metrics") -> JobSummary:
    return JobSummary(
        id="job-metrics",
        status="running",
        run_id=run_id,
        config_resource_id="config-resource",
        dataset_resource_id="dataset-resource",
        config_digest="c" * 64,
        dataset_id="d" * 64,
        config_path="configs/training.json",
        dataset_path="datasets/btc",
        artifact_root="research",
        submitted_at="2026-07-26T00:00:00+00:00",
        owner_instance_id="owner",
    )


def _write_events(tmp_path: Path, *, seed: int = 3, suffix: str = "") -> Path:
    run = (
        tmp_path
        / "research"
        / ".staging"
        / "run-metrics"
        / "members"
        / "member-000"
        / "tensorboard"
        / f"seed-{seed}-ppo{suffix}"
    )
    writer = SummaryWriter(log_dir=run)
    writer.add_scalar("train/learning_rate", 1.2e-4, 100)
    writer.add_scalar("train/learning_rate", 1.0e-4, 200)
    writer.add_scalar("train/approx_kl", 0.01, 200)
    writer.add_scalar("trade_rl/reward_absolute_component_mean", 0.002, 200)
    writer.add_scalar("secret/internal", 999.0, 200)
    writer.close()
    return run


def test_reader_returns_only_allowlisted_sorted_scalars(tmp_path: Path) -> None:
    _write_events(tmp_path)
    reader = StudioTrainingMetricsReader(settings(tmp_path))
    job = _job(tmp_path)

    status = reader.status(job, seed=3)
    page = reader.scalars(
        job,
        seed=3,
        tags=("train/learning_rate", "train/approx_kl"),
        after_step=0,
        limit=512,
        generation=status.generation,
    )

    assert status.available
    assert status.available_seeds == (3,)
    assert "secret/internal" not in status.available_tags
    assert "trade_rl/reward_absolute_component_mean" in status.available_tags
    assert [point.step for point in page.series[0].points] == [100, 200]
    assert page.next_step == 200


def test_reader_merges_resumed_event_directories_and_uses_cursor(
    tmp_path: Path,
) -> None:
    _write_events(tmp_path)
    resumed = _write_events(tmp_path, suffix="_2")
    writer = SummaryWriter(log_dir=resumed)
    writer.add_scalar("train/learning_rate", 8.0e-5, 300)
    writer.close()
    reader = StudioTrainingMetricsReader(settings(tmp_path))

    page = reader.scalars(
        _job(tmp_path),
        seed=3,
        tags=("train/learning_rate",),
        after_step=200,
        limit=512,
        generation=None,
    )
    assert [point.step for point in page.series[0].points] == [300]


def test_generation_remains_stable_when_event_file_is_appended(tmp_path: Path) -> None:
    run = (
        tmp_path
        / "research"
        / ".staging"
        / "run-metrics"
        / "members"
        / "member-000"
        / "tensorboard"
        / "seed-3-ppo"
    )
    writer = SummaryWriter(log_dir=run)
    writer.add_scalar("train/learning_rate", 1.2e-4, 100)
    writer.flush()
    reader = StudioTrainingMetricsReader(settings(tmp_path))
    job = _job(tmp_path)
    initial = reader.status(job, seed=3)

    writer.add_scalar("train/learning_rate", 1.0e-4, 200)
    writer.flush()
    updated = reader.status(job, seed=3)
    page = reader.scalars(
        job,
        seed=3,
        tags=("train/learning_rate",),
        after_step=100,
        limit=512,
        generation=initial.generation,
    )
    writer.close()

    assert updated.generation == initial.generation
    assert not page.reset_required
    assert [point.step for point in page.series[0].points] == [200]


def test_reader_requests_reset_when_generation_changes(tmp_path: Path) -> None:
    _write_events(tmp_path)
    reader = StudioTrainingMetricsReader(settings(tmp_path))
    page = reader.scalars(
        _job(tmp_path),
        seed=3,
        tags=("train/learning_rate",),
        after_step=100,
        limit=512,
        generation="0" * 64,
    )
    assert page.reset_required
    assert page.next_step == 0


def test_reader_rejects_symlinked_artifact_root_before_resolution(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-research"
    real_root.mkdir()
    (tmp_path / "research").symlink_to(real_root, target_is_directory=True)
    reader = StudioTrainingMetricsReader(settings(tmp_path))

    with pytest.raises(ArtifactInvalid, match="symlink"):
        reader.status(_job(tmp_path), seed=3)


def test_reader_rejects_unknown_tags_and_symlinks(tmp_path: Path) -> None:
    _write_events(tmp_path)
    reader = StudioTrainingMetricsReader(settings(tmp_path))
    with pytest.raises(InvalidStudioRequest, match="unknown"):
        reader.scalars(
            _job(tmp_path),
            seed=3,
            tags=("secret/internal",),
            after_step=0,
            limit=512,
            generation=None,
        )

    member = (
        tmp_path / "research" / ".staging" / "run-metrics" / "members" / "member-000"
    )
    target = member / "tensorboard"
    moved = member / "real-tensorboard"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ArtifactInvalid, match="symlink"):
        reader.status(_job(tmp_path), seed=3)
