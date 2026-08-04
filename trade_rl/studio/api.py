"""FastAPI application for the loopback-only Trade RL Studio runtime."""

from __future__ import annotations

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse

from trade_rl.studio.behavior_cloning_progress import (
    BehaviorCloningProgressResponse,
    StudioBehaviorCloningProgressReader,
)
from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.checkpoint_evaluations import (
    CheckpointEvaluationsResponse,
    StudioCheckpointEvaluationReader,
)
from trade_rl.studio.comparison import compare_runs
from trade_rl.studio.contracts import (
    ConfigListResponse,
    DatasetListResponse,
    EvidenceReport,
    JobListResponse,
    JobLogResponse,
    JobSummary,
    RunComparison,
    RunListResponse,
    ServingMonitorReport,
    StudioOverview,
    TrainingJobRequest,
)
from trade_rl.studio.errors import (
    ArtifactInvalid,
    IdentityConflict,
    InvalidStudioRequest,
    JobOwnershipLost,
    ResourceNotFound,
    StudioError,
)
from trade_rl.studio.evidence import inspect_run_evidence
from trade_rl.studio.jobs import JobSupervisor
from trade_rl.studio.serving_monitor import inspect_serving
from trade_rl.studio.settings import StudioSettings
from trade_rl.studio.supervised_jobs import SupervisedJobCatalog
from trade_rl.studio.telemetry import (
    StudioTelemetryReader,
    TelemetryEventsResponse,
    TelemetryStatusResponse,
)
from trade_rl.studio.training_metrics import (
    StudioTrainingMetricsReader,
    TrainingMetricsResponse,
    TrainingMetricsStatusResponse,
)

_ERROR_STATUS: tuple[tuple[type[StudioError], int], ...] = (
    (ResourceNotFound, 404),
    (InvalidStudioRequest, 400),
    (ArtifactInvalid, 422),
    (IdentityConflict, 409),
    (JobOwnershipLost, 409),
)


def _status_for(error: StudioError) -> int:
    for error_type, status_code in _ERROR_STATUS:
        if isinstance(error, error_type):
            return status_code
    return 500


def create_app(
    settings: StudioSettings,
    *,
    catalog: StudioCatalog | None = None,
    supervisor: JobSupervisor | None = None,
) -> FastAPI:
    """Construct an app with explicit filesystem dependencies for testing."""

    resolved_catalog = catalog or StudioCatalog(settings)
    resolved_supervisor = supervisor or JobSupervisor(
        settings,
        catalog=resolved_catalog,
    )
    telemetry_reader = StudioTelemetryReader(settings)
    training_metrics_reader = StudioTrainingMetricsReader(settings)
    checkpoint_reader = StudioCheckpointEvaluationReader(settings)
    behavior_cloning_reader = StudioBehaviorCloningProgressReader(settings)
    supervised_jobs = SupervisedJobCatalog(settings)

    def all_jobs() -> tuple[JobSummary, ...]:
        internal = resolved_supervisor.list_jobs()
        known = {job.id for job in internal}
        return internal + tuple(
            job for job in supervised_jobs.list() if job.id not in known
        )

    def resolve_job(job_id: str) -> JobSummary:
        try:
            return resolved_supervisor.get_job(job_id)
        except ResourceNotFound:
            return supervised_jobs.get(job_id)

    app = FastAPI(
        title="Trade RL Studio API",
        version="0.3.0",
        description="Loopback-only research artifact, job and replay telemetry API.",
    )

    @app.exception_handler(StudioError)
    async def studio_error_handler(
        _request: Request, error: StudioError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=_status_for(error),
            content={"detail": {"code": error.code, "message": str(error)}},
        )

    @app.get("/api/studio/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "productionStatus": "NO-GO"}

    @app.get("/api/studio/overview", response_model=StudioOverview)
    def overview() -> StudioOverview:
        return resolved_catalog.overview(all_jobs())

    @app.get("/api/studio/datasets", response_model=DatasetListResponse)
    def datasets() -> DatasetListResponse:
        items = resolved_catalog.list_datasets()
        return DatasetListResponse(
            items=items,
            total=len(items),
            invalid=sum(item.status == "INVALID" for item in items),
        )

    @app.get("/api/studio/runs", response_model=RunListResponse)
    def runs() -> RunListResponse:
        items = resolved_catalog.list_runs()
        return RunListResponse(
            items=items,
            total=len(items),
            invalid=sum(item.status == "INVALID" for item in items),
        )

    @app.get("/api/studio/configs", response_model=ConfigListResponse)
    def configs() -> ConfigListResponse:
        items = resolved_catalog.list_configs()
        return ConfigListResponse(
            items=items,
            total=len(items),
            invalid=sum(item.status == "INVALID" for item in items),
        )

    @app.get("/api/studio/compare", response_model=RunComparison)
    def comparison(
        left_resource_id: str,
        right_resource_id: str,
    ) -> RunComparison:
        left = resolved_catalog.resolve_run(left_resource_id)
        right = resolved_catalog.resolve_run(right_resource_id)
        try:
            return compare_runs(left, right)
        except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
            raise ArtifactInvalid(str(error)) from error

    @app.get(
        "/api/studio/runs/{run_resource_id}/evidence",
        response_model=EvidenceReport,
    )
    def evidence(run_resource_id: str) -> EvidenceReport:
        path = resolved_catalog.resolve_run_for_evidence(run_resource_id)
        return inspect_run_evidence(path, run_resource_id=run_resource_id)

    @app.get("/api/studio/serving", response_model=ServingMonitorReport)
    def serving() -> ServingMonitorReport:
        return inspect_serving(settings)

    @app.get("/api/studio/jobs", response_model=JobListResponse)
    def jobs() -> JobListResponse:
        items = all_jobs()
        return JobListResponse(items=items, total=len(items))

    @app.get("/api/studio/jobs/{job_id}", response_model=JobSummary)
    def job(job_id: str) -> JobSummary:
        return resolve_job(job_id)

    @app.get("/api/studio/jobs/{job_id}/log", response_model=JobLogResponse)
    def job_log(
        job_id: str,
        limit: int = Query(default=200, ge=1, le=2_000),
    ) -> JobLogResponse:
        resolved = resolve_job(job_id)
        if resolved.owner_instance_id == "external-supervisor":
            return JobLogResponse(job_id=job_id, lines=(), truncated=False)
        lines, truncated = resolved_supervisor.tail_log(job_id, limit=limit)
        return JobLogResponse(job_id=job_id, lines=lines, truncated=truncated)

    @app.get(
        "/api/studio/jobs/{job_id}/behavior-cloning/progress",
        response_model=BehaviorCloningProgressResponse,
    )
    def behavior_cloning_progress(job_id: str) -> BehaviorCloningProgressResponse:
        return behavior_cloning_reader.inspect(resolve_job(job_id))

    @app.get(
        "/api/studio/jobs/{job_id}/telemetry/status",
        response_model=TelemetryStatusResponse,
    )
    def telemetry_status(
        job_id: str,
        seed: int | None = Query(default=None, ge=0),
    ) -> TelemetryStatusResponse:
        return telemetry_reader.status(
            resolve_job(job_id),
            seed=seed,
        )

    @app.get(
        "/api/studio/jobs/{job_id}/telemetry/events",
        response_model=TelemetryEventsResponse,
    )
    def telemetry_events(
        job_id: str,
        seed: int | None = Query(default=None, ge=0),
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=512, ge=1, le=2_000),
        stream_generation: str | None = Query(
            default=None,
            pattern=(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}$"
            ),
        ),
    ) -> TelemetryEventsResponse:
        return telemetry_reader.events(
            resolve_job(job_id),
            seed=seed,
            after_sequence=after_sequence,
            limit=limit,
            stream_generation=stream_generation,
        )

    @app.get(
        "/api/studio/jobs/{job_id}/training-metrics/status",
        response_model=TrainingMetricsStatusResponse,
    )
    def training_metrics_status(
        job_id: str,
        seed: int | None = Query(default=None, ge=0),
    ) -> TrainingMetricsStatusResponse:
        return training_metrics_reader.status(
            resolve_job(job_id),
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
            resolve_job(job_id),
            seed=seed,
            tags=tuple(tag),
            after_step=after_step,
            limit=limit,
            generation=generation,
        )

    @app.get(
        "/api/studio/jobs/{job_id}/checkpoint-evaluations",
        response_model=CheckpointEvaluationsResponse,
    )
    def checkpoint_evaluations(job_id: str) -> CheckpointEvaluationsResponse:
        return checkpoint_reader.inspect(resolve_job(job_id))

    @app.post(
        "/api/studio/jobs/training",
        response_model=JobSummary,
        status_code=status.HTTP_201_CREATED,
    )
    def submit_training(request: TrainingJobRequest) -> JobSummary:
        return resolved_supervisor.submit_training(request)

    @app.post("/api/studio/jobs/{job_id}/cancel", response_model=JobSummary)
    def cancel(job_id: str) -> JobSummary:
        if resolve_job(job_id).owner_instance_id == "external-supervisor":
            raise InvalidStudioRequest("externally supervised jobs are read-only")
        return resolved_supervisor.cancel(job_id)

    return app
