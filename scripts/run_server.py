"""Start the authenticated read-only Trade RL serving plane."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI

from mars_lite.server.signal_server import create_app
from mars_lite.serving.audit_store import AuditStore
from mars_lite.serving.feature_provider import CsvFeatureProvider
from mars_lite.serving.registry import ModelRegistry
from mars_lite.serving.runtime import ServingRuntime


def build_app_from_env(environ: Mapping[str, str] | None = None) -> FastAPI:
    env = os.environ if environ is None else environ
    token = env.get("TRADE_RL_SERVING_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TRADE_RL_SERVING_TOKEN is required")
    release_git_sha = env.get("TRADE_RL_RELEASE_GIT_SHA", "").strip()
    if not release_git_sha:
        raise RuntimeError(
            "TRADE_RL_RELEASE_GIT_SHA is required for Production serving"
        )
    registry_dir = Path(env.get("TRADE_RL_REGISTRY_DIR", "output/model_registry"))
    audit_db = Path(env.get("TRADE_RL_AUDIT_DB", "output/serving/audit.sqlite3"))
    data_dir = Path(env.get("TRADE_RL_DATA_DIR", "data"))
    origins = tuple(
        origin.strip()
        for origin in env.get("TRADE_RL_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    )
    registry = ModelRegistry(registry_dir)
    runtime = ServingRuntime(
        registry=registry,
        audit_store=AuditStore(audit_db),
        release_git_sha=release_git_sha,
        strict_release_binding=True,
    )
    runtime.refresh()
    provider = CsvFeatureProvider(runtime=runtime, data_dir=data_dir)
    return create_app(
        runtime=runtime,
        feature_provider=provider,
        auth_token=token,
        allowed_origins=origins,
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("TRADE_RL_HOST", "127.0.0.1")
    port = int(os.environ.get("TRADE_RL_PORT", "8001"))
    uvicorn.run(build_app_from_env(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
