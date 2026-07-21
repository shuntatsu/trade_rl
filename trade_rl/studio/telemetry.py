"""Fail-closed Studio access to training telemetry artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from trade_rl.studio.contracts import JobSummary, StudioModel
from trade_rl.studio.errors import ArtifactInvalid
from trade_rl.studio.settings import StudioSettings
from trade_rl.telemetry.training import (
    TrainingTelemetryRecord,
    read_training_telemetry,
    training_telemetry_status,
)

_TELEMETRY_NAME = "training-telemetry.jsonl"


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
    record_count: int = Field(ge=0)
    last_sequence: int = Field(ge=0)
    malformed_lines: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    source: str | None = None


class TelemetryEventsResponse(StudioModel):
    items: tuple[TelemetryRecordResponse, ...]
    next_sequence: int = Field(ge=0)
    truncated: bool
    malformed_lines: int = Field(ge=0)
    sequence_gaps: tuple[tuple[int, int], ...]


def _response(record: TrainingTelemetryRecord) -> TelemetryRecordResponse:
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

    def _path(self, job: JobSummary) -> Path | None:
        root = self._artifact_root(job)
        candidates: list[Path] = []
        for namespace in (".staging", "runs", "failed"):
            run_root = (root / namespace / job.run_id).resolve()
            try:
                run_root.relative_to(root)
            except ValueError as error:
                raise ArtifactInvalid(
                    "telemetry run path escapes artifact root"
                ) from error
            if run_root.is_dir():
                candidates.extend(sorted(run_root.rglob(_TELEMETRY_NAME)))
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ArtifactInvalid("telemetry file escapes artifact root") from error
            if resolved.is_file() and not resolved.is_symlink():
                return resolved
        return None

    def _source(self, path: Path) -> str:
        try:
            return path.relative_to(self.settings.project_root.resolve()).as_posix()
        except ValueError as error:
            raise ArtifactInvalid("telemetry source is outside the project") from error

    def status(self, job: JobSummary) -> TelemetryStatusResponse:
        path = self._path(job)
        if path is None:
            return TelemetryStatusResponse(
                available=False,
                record_count=0,
                last_sequence=0,
                malformed_lines=0,
                size_bytes=0,
                source=None,
            )
        status = training_telemetry_status(path)
        return TelemetryStatusResponse(
            available=status.available,
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
        after_sequence: int,
        limit: int,
    ) -> TelemetryEventsResponse:
        path = self._path(job)
        if path is None:
            return TelemetryEventsResponse(
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
        return TelemetryEventsResponse(
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
