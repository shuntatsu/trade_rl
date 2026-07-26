"""Fail-closed access to allowlisted TensorBoard scalar artifacts."""

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
_RUN_DIRECTORY = re.compile(r"^seed-(?P<seed>\d+)-[a-z0-9_-]+(?:_\d+)?$")
_EVENT_PREFIX = "events.out.tfevents."
MetricGroup = Literal["optimization", "policy", "value", "trading"]
MetricUnit = Literal["raw", "rate", "percent", "currency"]

_METRICS: dict[str, tuple[str, MetricGroup, MetricUnit]] = {
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

    @staticmethod
    def _reject_symlink_chain(path: Path, *, stop: Path) -> None:
        try:
            relative = path.relative_to(stop)
        except ValueError as error:
            raise ArtifactInvalid(
                "TensorBoard artifact path is outside its root"
            ) from error
        if any(part == ".." for part in relative.parts):
            raise ArtifactInvalid("TensorBoard artifact path escapes its root")
        current = stop
        if current.is_symlink():
            raise ArtifactInvalid("TensorBoard artifact root is a symlink")
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ArtifactInvalid("TensorBoard artifact contains a symlink")

    def _artifact_root(self, job: JobSummary) -> Path:
        project_root = self.settings.project_root.resolve()
        relative = Path(job.artifact_root)
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise ArtifactInvalid("job artifact root escapes the Studio project")
        candidate = project_root / relative
        self._reject_symlink_chain(candidate, stop=project_root)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as error:
            raise ArtifactInvalid(
                "job artifact root escapes the Studio project"
            ) from error
        return resolved

    def _sources(self, job: JobSummary) -> dict[int, _SeedSource]:
        artifact_root = self._artifact_root(job)
        collected: dict[int, tuple[Path, list[Path], list[Path]]] = {}
        for namespace in (".staging", "runs", "failed"):
            raw_run_root = artifact_root / namespace / job.run_id
            self._reject_symlink_chain(raw_run_root, stop=artifact_root)
            run_root = raw_run_root.resolve()
            try:
                run_root.relative_to(artifact_root)
            except ValueError as error:
                raise ArtifactInvalid(
                    "training metric run path escapes artifact root"
                ) from error
            if not run_root.is_dir():
                continue
            tensorboard_roots = sorted(run_root.glob("members/member-*/tensorboard"))
            for tensorboard_root in tensorboard_roots:
                self._reject_symlink_chain(tensorboard_root, stop=run_root)
                resolved_tb = tensorboard_root.resolve()
                try:
                    resolved_tb.relative_to(run_root)
                except ValueError as error:
                    raise ArtifactInvalid(
                        "TensorBoard directory escapes run root"
                    ) from error
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
                            raise ArtifactInvalid(
                                "TensorBoard event file escapes run root"
                            ) from error
                        if resolved.is_file():
                            events.append(resolved)
                    if not events:
                        continue
                    existing = collected.get(seed)
                    if existing is not None and existing[0] != member_root:
                        raise ArtifactInvalid(
                            f"multiple ensemble members claim seed {seed}"
                        )
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
        """Identify an event-file set without changing when an event file grows."""

        return content_digest(
            {
                "event_files": tuple(
                    event.relative_to(source.member_root).as_posix()
                    for event in source.event_files
                ),
                "run_directories": tuple(
                    directory.relative_to(source.member_root).as_posix()
                    for directory in source.run_directories
                ),
                "schema_version": "studio_training_metrics_generation_v2",
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
                        if (
                            step < 0
                            or not math.isfinite(wall_time)
                            or not math.isfinite(value)
                        ):
                            raise ArtifactInvalid(
                                "TensorBoard scalar contains invalid values"
                            )
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
            raise ArtifactInvalid(
                "TensorBoard source is outside the project"
            ) from error

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
            selected = tuple(
                point for point in loaded.get(tag, ()) if point.step > after_step
            )[:limit]
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
