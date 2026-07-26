from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"guarded replacement failed for {relative}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def apply() -> None:
    _write(
        "trade_rl/studio/training_metrics.py",
        '''"""Fail-closed access to allowlisted TensorBoard scalar artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from trade_rl.artifacts.hashing import content_digest
from trade_rl.studio.contracts import JobSummary, StudioModel
from trade_rl.studio.errors import ArtifactInvalid, InvalidStudioRequest
from trade_rl.studio.settings import StudioSettings

_MAX_TAGS = 8
_MAX_POINTS = 2_000
_RUN_DIRECTORY = re.compile(r"^seed-(?P<seed>\\d+)-[a-z0-9_-]+(?:_\\d+)?$")
_EVENT_PREFIX = "events.out.tfevents."
_METRICS: dict[str, tuple[str, str, str]] = {
    "train/learning_rate": ("Learning rate", "optimization", "rate"),
    "train/loss": ("Total loss", "optimization", "raw"),
    "train/policy_gradient_loss": ("Policy gradient loss", "policy", "raw"),
    "train/value_loss": ("Value loss", "value", "raw"),
    "train/entropy_loss": ("Entropy loss", "policy", "raw"),
    "train/approx_kl": ("Approx KL", "policy", "raw"),
    "train/clip_fraction": ("Clip fraction", "policy", "percent"),
    "train/explained_variance": ("Explained variance", "value", "raw"),
    "trade_rl/reward_mean": ("Mean reward", "trading", "raw"),
    "trade_rl/portfolio_value_mean": ("Mean portfolio value", "trading", "currency"),
    "trade_rl/drawdown_mean": ("Mean drawdown", "trading", "percent"),
    "trade_rl/interval_cost_mean": ("Mean interval cost", "trading", "currency"),
    "trade_rl/action_abs_mean": ("Mean absolute action", "trading", "raw"),
    "trade_rl/action_abs_max": ("Maximum absolute action", "trading", "raw"),
}


class TrainingMetricPoint(StudioModel):
    step: int = Field(ge=0)
    wall_time: float
    value: float


class TrainingMetricSeries(StudioModel):
    tag: str
    display_name: str
    group: Literal["optimization", "policy", "value", "trading"]
    unit: Literal["raw", "rate", "percent", "currency"]
    points: tuple[TrainingMetricPoint, ...]


class TrainingMetricsStatusResponse(StudioModel):
    available: bool
    selected_seed: int | None = Field(default=None, ge=0)
    available_seeds: tuple[int, ...]
    available_tags: tuple[str, ...]
    last_step: int = Field(ge=0)
    source: str | None = None
    generation: str | None = None


class TrainingMetricsResponse(StudioModel):
    seed: int | None = Field(default=None, ge=0)
    series: tuple[TrainingMetricSeries, ...]
    next_step: int = Field(ge=0)
    generation: str | None = None
    reset_required: bool = False


@dataclass(frozen=True, slots=True)
class _SeedSource:
    member_root: Path
    run_directories: tuple[Path, ...]
    event_files: tuple[Path, ...]


class StudioTrainingMetricsReader:
    """Read finite scalar data only beneath the selected job and seed."""

    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings

    def _artifact_root(self, job: JobSummary) -> Path:
        project_root = self.settings.project_root.resolve()
        candidate = (project_root / job.artifact_root).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ArtifactInvalid("job artifact root escapes the Studio project") from error
        return candidate

    @staticmethod
    def _reject_symlink_chain(path: Path, *, stop: Path) -> None:
        current = path
        while current != stop:
            if current.is_symlink():
                raise ArtifactInvalid("TensorBoard artifact contains a symlink")
            current = current.parent
        if stop.is_symlink():
            raise ArtifactInvalid("TensorBoard artifact root is a symlink")

    def _sources(self, job: JobSummary) -> dict[int, _SeedSource]:
        artifact_root = self._artifact_root(job)
        collected: dict[int, tuple[Path, list[Path], list[Path]]] = {}
        for namespace in (".staging", "runs", "failed"):
            run_root = (artifact_root / namespace / job.run_id).resolve()
            try:
                run_root.relative_to(artifact_root)
            except ValueError as error:
                raise ArtifactInvalid("training metric run path escapes artifact root") from error
            if not run_root.is_dir():
                continue
            self._reject_symlink_chain(run_root, stop=artifact_root)
            tensorboard_roots = sorted(run_root.glob("members/member-*/tensorboard"))
            for tensorboard_root in tensorboard_roots:
                resolved_tb = tensorboard_root.resolve()
                try:
                    resolved_tb.relative_to(run_root)
                except ValueError as error:
                    raise ArtifactInvalid("TensorBoard directory escapes run root") from error
                self._reject_symlink_chain(tensorboard_root, stop=run_root)
                member_root = tensorboard_root.parent.resolve()
                for candidate in sorted(tensorboard_root.iterdir()):
                    if not candidate.is_dir():
                        continue
                    match = _RUN_DIRECTORY.fullmatch(candidate.name)
                    if match is None:
                        continue
                    self._reject_symlink_chain(candidate, stop=run_root)
                    seed = int(match.group("seed"))
                    events: list[Path] = []
                    for event in sorted(candidate.glob(f"{_EVENT_PREFIX}*")):
                        self._reject_symlink_chain(event, stop=run_root)
                        resolved = event.resolve()
                        try:
                            resolved.relative_to(run_root)
                        except ValueError as error:
                            raise ArtifactInvalid("TensorBoard event file escapes run root") from error
                        if not resolved.is_file():
                            continue
                        events.append(resolved)
                    if not events:
                        continue
                    existing = collected.get(seed)
                    if existing is not None and existing[0] != member_root:
                        raise ArtifactInvalid(f"multiple ensemble members claim seed {seed}")
                    if existing is None:
                        collected[seed] = (member_root, [candidate.resolve()], events)
                    else:
                        existing[1].append(candidate.resolve())
                        existing[2].extend(events)
        return {
            seed: _SeedSource(
                member_root=member_root,
                run_directories=tuple(sorted(set(run_directories))),
                event_files=tuple(sorted(set(event_files))),
            )
            for seed, (member_root, run_directories, event_files) in collected.items()
        }

    @staticmethod
    def _generation(source: _SeedSource) -> str:
        return content_digest(
            {
                "files": tuple(
                    (
                        event.as_posix(),
                        event.stat().st_size,
                        event.stat().st_mtime_ns,
                    )
                    for event in source.event_files
                ),
                "schema_version": "studio_training_metrics_generation_v1",
            }
        )

    @staticmethod
    def _load(source: _SeedSource) -> dict[str, tuple[TrainingMetricPoint, ...]]:
        try:
            from tensorboard.backend.event_processing import event_accumulator
        except ImportError as error:
            raise ArtifactInvalid("TensorBoard support is not installed") from error
        points: dict[str, dict[int, TrainingMetricPoint]] = {
            tag: {} for tag in _METRICS
        }
        size_guidance = {
            event_accumulator.SCALARS: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0,
        }
        try:
            for event_file in source.event_files:
                accumulator = event_accumulator.EventAccumulator(
                    str(event_file), size_guidance=size_guidance
                )
                accumulator.Reload()
                scalar_tags = set(accumulator.Tags().get("scalars", ()))
                for tag in _METRICS:
                    if tag not in scalar_tags:
                        continue
                    for item in accumulator.Scalars(tag):
                        step = int(item.step)
                        wall_time = float(item.wall_time)
                        value = float(item.value)
                        if step < 0 or not math.isfinite(wall_time) or not math.isfinite(value):
                            raise ArtifactInvalid("TensorBoard scalar contains invalid values")
                        previous = points[tag].get(step)
                        if previous is None or wall_time >= previous.wall_time:
                            points[tag][step] = TrainingMetricPoint(
                                step=step,
                                wall_time=wall_time,
                                value=value,
                            )
        except ArtifactInvalid:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ArtifactInvalid("TensorBoard event artifact is malformed") from error
        return {
            tag: tuple(by_step[step] for step in sorted(by_step))
            for tag, by_step in points.items()
            if by_step
        }

    def _source_label(self, source: _SeedSource) -> str:
        project_root = self.settings.project_root.resolve()
        try:
            return source.member_root.relative_to(project_root).as_posix()
        except ValueError as error:
            raise ArtifactInvalid("TensorBoard source is outside the project") from error

    def status(
        self,
        job: JobSummary,
        *,
        seed: int | None,
    ) -> TrainingMetricsStatusResponse:
        sources = self._sources(job)
        available_seeds = tuple(sorted(sources))
        selected_seed = min(sources) if seed is None and sources else seed
        source = sources.get(selected_seed) if selected_seed is not None else None
        if source is None:
            return TrainingMetricsStatusResponse(
                available=False,
                selected_seed=None,
                available_seeds=available_seeds,
                available_tags=(),
                last_step=0,
            )
        loaded = self._load(source)
        available_tags = tuple(tag for tag in _METRICS if tag in loaded)
        last_step = max(
            (series[-1].step for series in loaded.values() if series),
            default=0,
        )
        return TrainingMetricsStatusResponse(
            available=bool(available_tags),
            selected_seed=selected_seed,
            available_seeds=available_seeds,
            available_tags=available_tags,
            last_step=last_step,
            source=self._source_label(source),
            generation=self._generation(source),
        )

    def scalars(
        self,
        job: JobSummary,
        *,
        seed: int | None,
        tags: tuple[str, ...],
        after_step: int,
        limit: int,
        generation: str | None,
    ) -> TrainingMetricsResponse:
        if not tags:
            raise InvalidStudioRequest("at least one metric tag is required")
        if len(tags) > _MAX_TAGS or len(set(tags)) != len(tags):
            raise InvalidStudioRequest("metric tags must be unique and limited to 8")
        unknown = tuple(tag for tag in tags if tag not in _METRICS)
        if unknown:
            raise InvalidStudioRequest(f"unknown training metric tag: {unknown[0]}")
        if after_step < 0 or not 1 <= limit <= _MAX_POINTS:
            raise InvalidStudioRequest("training metric cursor or limit is invalid")
        sources = self._sources(job)
        selected_seed = min(sources) if seed is None and sources else seed
        source = sources.get(selected_seed) if selected_seed is not None else None
        if source is None:
            return TrainingMetricsResponse(
                seed=None,
                series=(),
                next_step=after_step,
            )
        current_generation = self._generation(source)
        if generation is not None and generation != current_generation:
            return TrainingMetricsResponse(
                seed=selected_seed,
                series=(),
                next_step=0,
                generation=current_generation,
                reset_required=True,
            )
        loaded = self._load(source)
        response_series: list[TrainingMetricSeries] = []
        next_step = after_step
        for tag in tags:
            display_name, group, unit = _METRICS[tag]
            selected = tuple(point for point in loaded.get(tag, ()) if point.step > after_step)[:limit]
            if selected:
                next_step = max(next_step, selected[-1].step)
            response_series.append(
                TrainingMetricSeries(
                    tag=tag,
                    display_name=display_name,
                    group=group,
                    unit=unit,
                    points=selected,
                )
            )
        return TrainingMetricsResponse(
            seed=selected_seed,
            series=tuple(response_series),
            next_step=next_step,
            generation=current_generation,
        )


__all__ = [
    "StudioTrainingMetricsReader",
    "TrainingMetricPoint",
    "TrainingMetricSeries",
    "TrainingMetricsResponse",
    "TrainingMetricsStatusResponse",
]
''',
    )

    _replace_once(
        "trade_rl/studio/api.py",
        "from trade_rl.studio.telemetry import (\n",
        "from trade_rl.studio.training_metrics import (\n"
        "    StudioTrainingMetricsReader,\n"
        "    TrainingMetricsResponse,\n"
        "    TrainingMetricsStatusResponse,\n"
        ")\n"
        "from trade_rl.studio.telemetry import (\n",
    )
    _replace_once(
        "trade_rl/studio/api.py",
        "    telemetry_reader = StudioTelemetryReader(settings)\n",
        "    telemetry_reader = StudioTelemetryReader(settings)\n"
        "    training_metrics_reader = StudioTrainingMetricsReader(settings)\n",
    )
    _replace_once(
        "trade_rl/studio/api.py",
        '''    @app.get(
        "/api/studio/jobs/{job_id}/checkpoint-evaluations",
''',
        '''    @app.get(
        "/api/studio/jobs/{job_id}/training-metrics/status",
        response_model=TrainingMetricsStatusResponse,
    )
    def training_metrics_status(
        job_id: str,
        seed: int | None = Query(default=None, ge=0),
    ) -> TrainingMetricsStatusResponse:
        return training_metrics_reader.status(
            resolved_supervisor.get_job(job_id),
            seed=seed,
        )

    @app.get(
        "/api/studio/jobs/{job_id}/training-metrics/scalars",
        response_model=TrainingMetricsResponse,
    )
    def training_metric_scalars(
        job_id: str,
        tag: list[str] = Query(default=[]),
        seed: int | None = Query(default=None, ge=0),
        after_step: int = Query(default=0, ge=0),
        limit: int = Query(default=512, ge=1, le=2_000),
        generation: str | None = Query(default=None, min_length=64, max_length=64),
    ) -> TrainingMetricsResponse:
        return training_metrics_reader.scalars(
            resolved_supervisor.get_job(job_id),
            seed=seed,
            tags=tuple(tag),
            after_step=after_step,
            limit=limit,
            generation=generation,
        )

    @app.get(
        "/api/studio/jobs/{job_id}/checkpoint-evaluations",
''',
    )

    _write(
        "tests/studio/test_training_metrics.py",
        '''from __future__ import annotations

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
    assert [point.step for point in page.series[0].points] == [100, 200]
    assert page.next_step == 200


def test_reader_merges_resumed_event_directories_and_uses_cursor(tmp_path: Path) -> None:
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

    member = tmp_path / "research" / ".staging" / "run-metrics" / "members" / "member-000"
    target = member / "tensorboard"
    moved = member / "real-tensorboard"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ArtifactInvalid, match="symlink"):
        reader.status(_job(tmp_path), seed=3)
''',
    )

    _write(
        "tests/studio/test_training_metrics_api.py",
        '''from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .test_api import client
from .test_jobs import request


def test_training_metrics_endpoints_return_status_and_allowlisted_scalars(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="run-metrics").model_dump(by_alias=True),
    )
    job_id = created.json()["id"]
    run = (
        tmp_path
        / "research"
        / ".staging"
        / "run-metrics"
        / "members"
        / "member-000"
        / "tensorboard"
        / "seed-7-ppo"
    )
    writer = SummaryWriter(log_dir=run)
    writer.add_scalar("train/learning_rate", 1.2e-4, 100)
    writer.add_scalar("secret/internal", 999.0, 100)
    writer.close()

    status = api.get(
        f"/api/studio/jobs/{job_id}/training-metrics/status",
        params={"seed": 7},
    )
    assert status.status_code == 200
    assert status.json()["available"] is True
    assert status.json()["availableTags"] == ["train/learning_rate"]

    scalars = api.get(
        f"/api/studio/jobs/{job_id}/training-metrics/scalars",
        params=[("seed", "7"), ("tag", "train/learning_rate")],
    )
    assert scalars.status_code == 200
    assert scalars.json()["series"][0]["points"][0]["step"] == 100


def test_training_metrics_api_returns_empty_and_rejects_unknown_tags(
    tmp_path: Path,
) -> None:
    api, _, catalog, _ = client(tmp_path)
    created = api.post(
        "/api/studio/jobs/training",
        json=request(catalog, run_id="run-empty").model_dump(by_alias=True),
    )
    job_id = created.json()["id"]
    empty = api.get(f"/api/studio/jobs/{job_id}/training-metrics/status")
    assert empty.status_code == 200
    assert empty.json()["available"] is False

    unknown = api.get(
        f"/api/studio/jobs/{job_id}/training-metrics/scalars",
        params={"tag": "secret/internal"},
    )
    assert unknown.status_code == 400
    assert unknown.json()["detail"]["code"] == "invalid_request"
    assert api.get("/api/studio/jobs/missing/training-metrics/status").status_code == 404
''',
    )


if __name__ == "__main__":
    apply()
