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


class ComparisonMetric(StudioModel):
    key: str
    label: str
    left_value: float | None = None
    right_value: float | None = None
    delta: float | None = None
    preference: Literal["higher", "lower", "neutral"] = "neutral"


class ConfigDifference(StudioModel):
    path: str
    left: str | None = None
    right: str | None = None


class FoldComparison(StudioModel):
    label: str
    left_selected_return: float | None = None
    left_baseline_return: float | None = None
    right_selected_return: float | None = None
    right_baseline_return: float | None = None


class ComparisonSeriesPoint(StudioModel):
    label: str
    left: float | None = None
    right: float | None = None
    left_baseline: float | None = None
    right_baseline: float | None = None


class RunComparison(StudioModel):
    left_run_id: str
    right_run_id: str
    metrics: tuple[ComparisonMetric, ...]
    config_differences: tuple[ConfigDifference, ...]
    folds: tuple[FoldComparison, ...]
    wealth: tuple[ComparisonSeriesPoint, ...]
    production_status: Literal["NO-GO"] = "NO-GO"


class EvidenceNode(StudioModel):
    key: str
    label: str
    status: Literal["VERIFIED", "PRESENT", "ABSENT", "INVALID"]
    required: bool
    digest: str | None = None
    path: str | None = None
    detail: str


class FileIntegritySummary(StudioModel):
    status: Literal["VERIFIED", "INVALID"]
    declared_count: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    total_size_bytes: int = Field(ge=0)


class EvidenceReport(StudioModel):
    run_id: str
    run_kind: str
    status: Literal["VALID", "INVALID"]
    production_status: Literal["NO-GO"] = "NO-GO"
    nodes: tuple[EvidenceNode, ...]
    files: FileIntegritySummary
    validation_error: str | None = None


class ServingCheck(StudioModel):
    key: str
    label: str
    status: Literal["PASS", "WARN", "FAIL"]
    detail: str


class PaperInferenceSnapshot(StudioModel):
    recorded_at: str
    bundle_digest: str
    dataset_id: str
    decision_index: int = Field(ge=0)
    target_weights: dict[str, float]
    latency_ms: float = Field(ge=0.0)
    snapshot_digest: str


class ServingMonitorReport(StudioModel):
    state: Literal["IDLE", "VALID", "INVALID"]
    production_status: Literal["NO-GO"] = "NO-GO"
    active_bundle_digest: str | None = None
    dataset_id: str | None = None
    run_kind: str | None = None
    policy_digest: str | None = None
    action_schema: str | None = None
    observation_schema: str | None = None
    release_attestation_present: bool = False
    checks: tuple[ServingCheck, ...]
    paper_snapshot: PaperInferenceSnapshot | None = None
    validation_error: str | None = None


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
