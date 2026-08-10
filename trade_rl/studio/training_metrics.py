"""Fail-closed access to allowlisted TensorBoard scalar artifacts."""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol

from pydantic import Field

from trade_rl.artifacts.hashing import content_digest
from trade_rl.studio.contracts import JobSummary, StudioModel
from trade_rl.studio.errors import ArtifactInvalid, InvalidStudioRequest
from trade_rl.studio.settings import StudioSettings

_MAX_TAGS = 8
_MAX_POINTS = 2_000
_DEFAULT_MAX_CACHED_SOURCES = 32
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
    "trade_rl/baseline_portfolio_value_mean": (
        "Mean baseline portfolio value",
        "trading",
        "currency",
    ),
    "trade_rl/drawdown_mean": ("Mean drawdown", "trading", "percent"),
    "trade_rl/interval_cost_mean": ("Mean interval cost", "trading", "currency"),
    "trade_rl/reward_growth_raw_mean": ("Mean raw reward growth", "trading", "raw"),
    "trade_rl/reward_absolute_component_mean": (
        "Mean absolute reward component",
        "trading",
        "raw",
    ),
    "trade_rl/reward_excess_component_mean": (
        "Mean excess reward component",
        "trading",
        "raw",
    ),
    "trade_rl/reward_baseline_penalty_weighted_mean": (
        "Mean baseline reward penalty",
        "trading",
        "raw",
    ),
    "trade_rl/reward_drawdown_penalty_weighted_mean": (
        "Mean drawdown reward penalty",
        "trading",
        "raw",
    ),
    "trade_rl/reward_projection_penalty_weighted_mean": (
        "Mean projection reward penalty",
        "trading",
        "raw",
    ),
    "trade_rl/reward_terminal_penalty_weighted_mean": (
        "Mean terminal reward penalty",
        "trading",
        "raw",
    ),
    "trade_rl/reward_margin_penalty_weighted_mean": (
        "Mean margin reward penalty",
        "trading",
        "raw",
    ),
    "trade_rl/reward_total_raw_mean": ("Mean raw total reward", "trading", "raw"),
    "trade_rl/rolling_growth_gap_mean": ("Mean rolling growth gap", "trading", "raw"),
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


class _ScalarEvent(Protocol):
    step: int
    wall_time: float
    value: float


class _EventAccumulator(Protocol):
    def Reload(self) -> object: ...

    def Tags(self) -> Mapping[str, Sequence[str]]: ...

    def Scalars(self, tag: str) -> Sequence[_ScalarEvent]: ...


_AccumulatorFactory = Callable[[Path], _EventAccumulator]
_MetricSnapshot = dict[str, tuple[TrainingMetricPoint, ...]]


@dataclass(frozen=True, slots=True)
class _SeedSource:
    member_root: Path
    run_directories: tuple[Path, ...]
    event_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _EventFingerprint:
    size: int
    modified_ns: int
    device: int
    inode: int

    @classmethod
    def from_path(cls, path: Path) -> _EventFingerprint:
        status = path.stat()
        return cls(
            size=int(status.st_size),
            modified_ns=int(status.st_mtime_ns),
            device=int(status.st_dev),
            inode=int(status.st_ino),
        )


@dataclass(frozen=True, slots=True)
class _EventFileSnapshot:
    fingerprint: _EventFingerprint
    accumulator: _EventAccumulator
    points: _MetricSnapshot


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    source: _SeedSource
    generation: str
    files: Mapping[Path, _EventFileSnapshot]
    merged: _MetricSnapshot


class StudioTrainingMetricsReader:
    """Read finite scalar data only beneath the selected job and seed."""

    def __init__(
        self,
        settings: StudioSettings,
        *,
        accumulator_factory: _AccumulatorFactory | None = None,
        max_cached_sources: int = _DEFAULT_MAX_CACHED_SOURCES,
    ) -> None:
        if (
            isinstance(max_cached_sources, bool)
            or not isinstance(max_cached_sources, int)
            or max_cached_sources <= 0
        ):
            raise ValueError("max_cached_sources must be a positive integer")
        self.settings = settings
        self._accumulator_factory = accumulator_factory or self._build_accumulator
        self._max_cached_sources = max_cached_sources
        self._cache: OrderedDict[tuple[Path, str], _SourceSnapshot] = OrderedDict()
        self._cache_lock = RLock()

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
    def _build_accumulator(path: Path) -> _EventAccumulator:
        try:
            from tensorboard.backend.event_processing import event_accumulator
        except ImportError as error:
            raise ArtifactInvalid("TensorBoard support is not installed") from error
        size_guidance = {
            event_accumulator.SCALARS: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.TENSORS: 0,
        }
        return event_accumulator.EventAccumulator(
            str(path),
            size_guidance=size_guidance,
        )

    @staticmethod
    def _append_compatible(
        previous: _EventFingerprint,
        current: _EventFingerprint,
    ) -> bool:
        stable_identity = (
            previous.inode != 0
            and current.inode != 0
            and previous.device == current.device
            and previous.inode == current.inode
        )
        return stable_identity and current.size >= previous.size

    @staticmethod
    def _points_from_accumulator(accumulator: _EventAccumulator) -> _MetricSnapshot:
        scalar_tags = set(accumulator.Tags().get("scalars", ()))
        points: dict[str, dict[int, TrainingMetricPoint]] = {
            tag: {} for tag in _METRICS
        }
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
        return {
            tag: tuple(by_step[step] for step in sorted(by_step))
            for tag, by_step in points.items()
            if by_step
        }

    @staticmethod
    def _merge_files(
        source: _SeedSource,
        files: Mapping[Path, _EventFileSnapshot],
    ) -> _MetricSnapshot:
        points: dict[str, dict[int, TrainingMetricPoint]] = {
            tag: {} for tag in _METRICS
        }
        for event_file in source.event_files:
            for tag, series in files[event_file].points.items():
                for point in series:
                    previous = points[tag].get(point.step)
                    if previous is None or point.wall_time >= previous.wall_time:
                        points[tag][point.step] = point
        return {
            tag: tuple(by_step[step] for step in sorted(by_step))
            for tag, by_step in points.items()
            if by_step
        }

    def _publish_cache(
        self,
        key: tuple[Path, str],
        snapshot: _SourceSnapshot,
    ) -> None:
        self._cache[key] = snapshot
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cached_sources:
            self._cache.popitem(last=False)

    def _load(self, source: _SeedSource) -> _MetricSnapshot:
        generation = self._generation(source)
        key = (source.member_root, generation)
        with self._cache_lock:
            cached = self._cache.get(key)
            try:
                fingerprints = {
                    event_file: _EventFingerprint.from_path(event_file)
                    for event_file in source.event_files
                }
                if cached is not None and all(
                    cached.files[event_file].fingerprint == fingerprints[event_file]
                    for event_file in source.event_files
                ):
                    self._cache.move_to_end(key)
                    return cached.merged

                files: dict[Path, _EventFileSnapshot] = {}
                for event_file in source.event_files:
                    fingerprint = fingerprints[event_file]
                    previous = None if cached is None else cached.files.get(event_file)
                    if previous is not None and previous.fingerprint == fingerprint:
                        files[event_file] = previous
                        continue
                    accumulator = (
                        previous.accumulator
                        if previous is not None
                        and self._append_compatible(previous.fingerprint, fingerprint)
                        else self._accumulator_factory(event_file)
                    )
                    accumulator.Reload()
                    files[event_file] = _EventFileSnapshot(
                        fingerprint=fingerprint,
                        accumulator=accumulator,
                        points=self._points_from_accumulator(accumulator),
                    )
                snapshot = _SourceSnapshot(
                    source=source,
                    generation=generation,
                    files=files,
                    merged=self._merge_files(source, files),
                )
            except ArtifactInvalid:
                self._cache.pop(key, None)
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._cache.pop(key, None)
                raise ArtifactInvalid(
                    "TensorBoard event artifact is malformed"
                ) from error
            self._publish_cache(key, snapshot)
            return snapshot.merged

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
