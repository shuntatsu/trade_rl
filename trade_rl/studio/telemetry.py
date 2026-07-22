"""Fail-closed Studio access to training telemetry artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from trade_rl.studio.contracts import JobSummary, StudioModel
from trade_rl.studio.errors import ArtifactInvalid
from trade_rl.studio.settings import StudioSettings
from trade_rl.telemetry import (
    read_training_telemetry,
    training_telemetry_status,
)

if TYPE_CHECKING:
    from trade_rl.telemetry.training import (
        TrainingTelemetryRecord as TelemetryRecordInput,
    )

_TELEMETRY_NAME = "training-telemetry.jsonl"
_SEED_DIRECTORY = re.compile(r"^seed-(\d+)$")


class TelemetryRecordResponse(StudioModel):
    schema_version: Literal["training_telemetry_v1"] = "training_telemetry_v1"
    sequence: int = Field(ge=1)
    recorded_at: str
    global_step: int = Field(ge=0)
    environment_step: int = Field(ge=0)
    seed: int = Field(ge=0)
    environment_id: int = Field(ge=0)
    event_type: Literal[
        "rollout",
        "position",
        "risk",
        "episode_end",
        "checkpoint",
        "gap",
    ]
    market_index: int | None = Field(default=None, ge=0)
    market_time: str | None = None
    symbol: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    action: tuple[float, ...]
    executed_target: tuple[float, ...]
    weights_before: tuple[float, ...]
    weights_after: tuple[float, ...]
    portfolio_value: float | None = None
    baseline_portfolio_value: float | None = None
    reward: float | None = None
    drawdown: float | None = None
    interval_cost: float | None = None
    interval_return: float | None = None
    risk_reasons: tuple[str, ...]
    emergency_deleverage: bool
    terminated: bool
    truncated: bool


class TelemetryStatusResponse(StudioModel):
    available: bool
    selected_seed: int | None = Field(default=None, ge=0)
    available_seeds: tuple[int, ...]
    record_count: int = Field(ge=0)
    last_sequence: int = Field(ge=0)
    malformed_lines: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    source: str | None = None


class TelemetryEventsResponse(StudioModel):
    seed: int | None = Field(default=None, ge=0)
    items: tuple[TelemetryRecordResponse, ...]
    next_sequence: int = Field(ge=0)
    truncated: bool
    malformed_lines: int = Field(ge=0)
    sequence_gaps: tuple[tuple[int, int], ...]


def _response(record: TelemetryRecordInput) -> TelemetryRecordResponse:
    return TelemetryRecordResponse.model_validate(record.to_json_dict())


class StudioTelemetryReader:
    """Resolve telemetry only beneath a job's declared artifact root and run."""

    def __init__(self, settings: StudioSettings) -> None:
        self.settings = settings

    def _artifact_root(self, job: JobSummary) -> Path:
        candidate = (self.settings.project_root / job.artifact_root).resolve()
        project_root = self.settings.project_root.resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ArtifactInvalid(
                "job artifact root escapes the Studio project"
            ) from error
        return candidate

    @staticmethod
    def _seed_from_path(path: Path, *, run_root: Path) -> int | None:
        relative = path.relative_to(run_root)
        for part in relative.parts:
            match = _SEED_DIRECTORY.fullmatch(part)
            if match is not None:
                return int(match.group(1))
        page = read_training_telemetry(path, limit=1)
        return page.items[0].seed if page.items else None

    def _paths(self, job: JobSummary) -> dict[int, Path]:
        root = self._artifact_root(job)
        streams: dict[int, Path] = {}
        for namespace in (".staging", "runs", "failed"):
            run_root = (root / namespace / job.run_id).resolve()
            try:
                run_root.relative_to(root)
            except ValueError as error:
                raise ArtifactInvalid(
                    "telemetry run path escapes artifact root"
                ) from error
            if not run_root.is_dir():
                continue
            for candidate in sorted(run_root.rglob(_TELEMETRY_NAME)):
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(run_root)
                except ValueError as error:
                    raise ArtifactInvalid(
                        "telemetry file escapes artifact root"
                    ) from error
                if not resolved.is_file() or resolved.is_symlink():
                    continue
                seed = self._seed_from_path(resolved, run_root=run_root)
                if seed is not None:
                    streams.setdefault(seed, resolved)
        return streams

    def _selection(
        self,
        job: JobSummary,
        seed: int | None,
    ) -> tuple[dict[int, Path], int | None, Path | None]:
        streams = self._paths(job)
        if seed is None:
            selected_seed = min(streams) if streams else None
        else:
            selected_seed = seed if seed in streams else None
        return (
            streams,
            selected_seed,
            (None if selected_seed is None else streams[selected_seed]),
        )

    def _source(self, path: Path) -> str:
        try:
            return path.relative_to(self.settings.project_root.resolve()).as_posix()
        except ValueError as error:
            raise ArtifactInvalid("telemetry source is outside the project") from error

    def status(
        self,
        job: JobSummary,
        *,
        seed: int | None = None,
    ) -> TelemetryStatusResponse:
        streams, selected_seed, path = self._selection(job, seed)
        available_seeds = tuple(sorted(streams))
        if path is None:
            return TelemetryStatusResponse(
                available=False,
                selected_seed=None,
                available_seeds=available_seeds,
                record_count=0,
                last_sequence=0,
                malformed_lines=0,
                size_bytes=0,
                source=None,
            )
        status = training_telemetry_status(path)
        return TelemetryStatusResponse(
            available=status.available,
            selected_seed=selected_seed,
            available_seeds=available_seeds,
            record_count=status.record_count,
            last_sequence=status.last_sequence,
            malformed_lines=status.malformed_lines,
            size_bytes=status.size_bytes,
            source=self._source(path),
        )

    def events(
        self,
        job: JobSummary,
        *,
        seed: int | None = None,
        after_sequence: int,
        limit: int,
    ) -> TelemetryEventsResponse:
        _, selected_seed, path = self._selection(job, seed)
        if path is None:
            return TelemetryEventsResponse(
                seed=None,
                items=(),
                next_sequence=after_sequence,
                truncated=False,
                malformed_lines=0,
                sequence_gaps=(),
            )
        page = read_training_telemetry(
            path,
            after_sequence=after_sequence,
            limit=limit,
        )
        if selected_seed is None or any(
            item.seed != selected_seed for item in page.items
        ):
            raise ArtifactInvalid(
                "telemetry record seed does not match stream identity"
            )
        return TelemetryEventsResponse(
            seed=selected_seed,
            items=tuple(_response(item) for item in page.items),
            next_sequence=page.next_sequence,
            truncated=page.truncated,
            malformed_lines=page.malformed_lines,
            sequence_gaps=page.sequence_gaps,
        )


__all__ = [
    "StudioTelemetryReader",
    "TelemetryEventsResponse",
    "TelemetryRecordResponse",
    "TelemetryStatusResponse",
]
