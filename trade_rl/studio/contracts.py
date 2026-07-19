"""Typed HTTP and catalog contracts for Trade RL Studio."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class StudioModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class SystemMetric(StudioModel):
    label: str
    value: float = Field(ge=0.0, le=100.0)
    detail: str


class SystemSummary(StudioModel):
    gpu_name: str
    cuda_ready: bool
    python_version: str
    metrics: tuple[SystemMetric, ...]


class DatasetSummary(StudioModel):
    id: str
    name: str
    relative_path: str
    market: str
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    range: str
    status: Literal["VALID", "INVALID"]
    feature_count: int = Field(ge=0)
    bar_count: int = Field(ge=0)
    symbol_count: int = Field(ge=0)
    updated: str
    validation_error: str | None = None


class RunSummary(StudioModel):
    id: str
    relative_path: str
    run_kind: str
    algorithm: str
    dataset_id: str
    period: str
    created_at: str
    completed_at: str
    file_count: int = Field(ge=0)
    sharpe: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    production_status: Literal["NO-GO"] = "NO-GO"
    status: Literal["VALID", "INVALID"]
    validation_error: str | None = None


class ConfigSummary(StudioModel):
    name: str
    relative_path: str
    algorithm: str
    status: Literal["VALID", "INVALID"]
    validation_error: str | None = None


class JobSummary(StudioModel):
    id: str
    kind: Literal["training"] = "training"
    status: Literal[
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelling",
        "cancelled",
    ]
    run_id: str
    config_path: str
    dataset_path: str
    artifact_root: str
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None


class ActiveJob(StudioModel):
    id: str
    algorithm: str
    phase: str
    seed_progress: str
    progress: float = Field(ge=0.0, le=100.0)


class StudioAlert(StudioModel):
    level: Literal["warning", "info"]
    message: str
    age: str


class EquityPoint(StudioModel):
    label: str
    rl: float
    baseline: float


class StabilityFold(StudioModel):
    label: str
    low: float
    median: float
    high: float


class ProductionAssessment(StudioModel):
    status: Literal["NO-GO"] = "NO-GO"
    reasons: tuple[str, ...]


class StudioOverview(StudioModel):
    system: SystemSummary
    latest_dataset: DatasetSummary | None
    active_jobs: tuple[ActiveJob, ...]
    runs: tuple[RunSummary, ...]
    alerts: tuple[StudioAlert, ...]
    equity: tuple[EquityPoint, ...]
    stability: tuple[StabilityFold, ...]
    assessment: ProductionAssessment


class DatasetListResponse(StudioModel):
    items: tuple[DatasetSummary, ...]
    total: int = Field(ge=0)
    invalid: int = Field(ge=0)


class RunListResponse(StudioModel):
    items: tuple[RunSummary, ...]
    total: int = Field(ge=0)
    invalid: int = Field(ge=0)


class ConfigListResponse(StudioModel):
    items: tuple[ConfigSummary, ...]
    total: int = Field(ge=0)
    invalid: int = Field(ge=0)


class JobListResponse(StudioModel):
    items: tuple[JobSummary, ...]
    total: int = Field(ge=0)


class TrainingJobRequest(StudioModel):
    config_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    run_id: str = Field(min_length=1, max_length=128)
    artifact_root: str | None = None


class JobLogResponse(StudioModel):
    job_id: str
    lines: tuple[str, ...]
    truncated: bool
