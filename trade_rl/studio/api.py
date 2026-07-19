"""FastAPI application for the local-only Trade RL Studio runtime."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, status

from trade_rl.studio.catalog import StudioCatalog
from trade_rl.studio.comparison import compare_runs
from trade_rl.studio.evidence import inspect_run_evidence
from trade_rl.studio.serving_monitor import inspect_serving
from trade_rl.studio.contracts import (
    ConfigListResponse,
    EvidenceReport,
    DatasetListResponse,
    JobListResponse,
    JobLogResponse,
    JobSummary,
    RunComparison,
    RunListResponse,
    ServingMonitorReport,
    StudioOverview,
    TrainingJobRequest,
)
from trade_rl.studio.jobs import JobSupervisor
from trade_rl.studio.settings import StudioSettings


def _raise_http(error: Exception) -> None:
    if isinstance(error, KeyError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, FileExistsError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, RuntimeError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    raise error


def create_app(
    settings: StudioSettings,
    *,
    catalog: StudioCatalog | None = None,
    supervisor: JobSupervisor | None = None,
) -> FastAPI:
    """Construct an app with explicit filesystem dependencies for testing."""

    resolved_catalog = catalog or StudioCatalog(settings)
    resolved_supervisor = supervisor or JobSupervisor(settings)
    app = FastAPI(
        title="Trade RL Studio API",
        version="0.1.0",
        description="Local-only research artifact and exploratory job API.",
    )

    @app.get("/api/studio/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "productionStatus": "NO-GO"}

    @app.get("/api/studio/overview", response_model=StudioOverview)
    def overview() -> StudioOverview:
        return resolved_catalog.overview(resolved_supervisor.list_jobs())

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
    def comparison(left_run_id: str, right_run_id: str) -> RunComparison:
        try:
            left = resolved_catalog.resolve_run(left_run_id)
            right = resolved_catalog.resolve_run(right_run_id)
            return compare_runs(left, right)
        except Exception as error:
            _raise_http(error)
            raise AssertionError("unreachable")

    @app.get(
        "/api/studio/runs/{run_id}/evidence",
        response_model=EvidenceReport,
    )
    def evidence(run_id: str) -> EvidenceReport:
        try:
            return inspect_run_evidence(resolved_catalog.resolve_run_for_evidence(run_id))
        except Exception as error:
            _raise_http(error)
            raise AssertionError("unreachable")

    @app.get("/api/studio/serving", response_model=ServingMonitorReport)
    def serving() -> ServingMonitorReport:
        return inspect_serving(settings)

    @app.get("/api/studio/jobs", response_model=JobListResponse)
    def jobs() -> JobListResponse:
        items = resolved_supervisor.list_jobs()
        return JobListResponse(items=items, total=len(items))

    @app.get("/api/studio/jobs/{job_id}", response_model=JobSummary)
    def job(job_id: str) -> JobSummary:
        try:
            return resolved_supervisor.get_job(job_id)
        except Exception as error:
            _raise_http(error)
            raise AssertionError("unreachable")

    @app.get("/api/studio/jobs/{job_id}/log", response_model=JobLogResponse)
    def job_log(
        job_id: str,
        limit: int = Query(default=200, ge=1, le=2_000),
    ) -> JobLogResponse:
        try:
            lines, truncated = resolved_supervisor.tail_log(job_id, limit=limit)
            return JobLogResponse(job_id=job_id, lines=lines, truncated=truncated)
        except Exception as error:
            _raise_http(error)
            raise AssertionError("unreachable")

    @app.post(
        "/api/studio/jobs/training",
        response_model=JobSummary,
        status_code=status.HTTP_201_CREATED,
    )
    def submit_training(request: TrainingJobRequest) -> JobSummary:
        try:
            return resolved_supervisor.submit_training(request)
        except Exception as error:
            _raise_http(error)
            raise AssertionError("unreachable")

    @app.post("/api/studio/jobs/{job_id}/cancel", response_model=JobSummary)
    def cancel(job_id: str) -> JobSummary:
        try:
            return resolved_supervisor.cancel(job_id)
        except Exception as error:
            _raise_http(error)
            raise AssertionError("unreachable")

    return app
