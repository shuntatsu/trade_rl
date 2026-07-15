from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_training_compose_requests_gpu_and_uses_only_named_runtime_volume() -> None:
    compose = (ROOT / "compose.training.yaml").read_text(encoding="utf-8")

    assert "gpus: all" in compose
    assert "trade-rl-training-data:/workspace/var" in compose
    assert "trade-rl-training-data:" in compose
    assert "./:/workspace" not in compose


def test_training_container_runs_preflight_then_maintained_full_runner() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")
    compose = (ROOT / "compose.training.yaml").read_text(encoding="utf-8")

    assert "FROM python:3.12" in dockerfile
    assert "USER trainer" in dockerfile
    assert "examples/binance-multitimeframe/training_cuda_preflight.py" in compose
    assert "examples/binance-multitimeframe/run_full_research.py" in compose
    assert "--work-root /workspace/var/runs/$${TRADE_RL_RUN_GENERATION}" in compose
    assert "--cache-root /workspace/var/cache/binance-vision" in compose
    assert "TRADE_RL_RUN_GENERATION:?" in compose
    assert "TRADE_RL_RUN_GENERATION: ${TRADE_RL_RUN_GENERATION:-}" in compose


def test_training_docker_context_excludes_local_state() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    for entry in (".git", ".venv", ".worktrees", "var", "__pycache__"):
        assert entry in ignored


def test_training_image_requires_and_exports_packaged_git_provenance() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")
    compose = (ROOT / "compose.training.yaml").read_text(encoding="utf-8")

    for name in ("TRADE_RL_GIT_COMMIT", "TRADE_RL_GIT_DIRTY"):
        assert f"ARG {name}" in dockerfile
        assert f"{name}=${{{name}}}" in dockerfile
        assert f"{name}: ${{{name}:-}}" in compose
    assert "^[0-9a-f]{40}$" in dockerfile
    assert '"true"|"false"' in dockerfile


def test_training_compose_renders_without_host_provenance_or_generation() -> None:
    environment = os.environ.copy()
    for name in (
        "TRADE_RL_GIT_COMMIT",
        "TRADE_RL_GIT_DIRTY",
        "TRADE_RL_RUN_GENERATION",
    ):
        environment.pop(name, None)

    completed = subprocess.run(
        ["docker", "compose", "-f", "compose.training.yaml", "config"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert 'TRADE_RL_GIT_COMMIT: ""' in completed.stdout
    assert 'TRADE_RL_GIT_DIRTY: ""' in completed.stdout
    assert 'TRADE_RL_RUN_GENERATION: ""' in completed.stdout


def test_training_dockerfile_caches_dependencies_before_provenance_stamp() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")

    dependency_sync = dockerfile.index("uv sync --frozen --extra train-sb3 --no-dev")
    commit_argument = dockerfile.index("ARG TRADE_RL_GIT_COMMIT")
    source_copy = dockerfile.index("COPY trade_rl ./trade_rl")
    provenance_validation = dockerfile.index("^[0-9a-f]{40}$")

    assert "--no-install-project" in dockerfile
    assert dependency_sync < source_copy < commit_argument < provenance_validation


def test_training_runbook_uses_unique_generations_and_shared_cache() -> None:
    runbook = (ROOT / "docs/operations/docker-gpu-full-training.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(runbook.split())

    assert "$env:TRADE_RL_RUN_GENERATION" in runbook
    assert "/workspace/var/runs/$TRADE_RL_RUN_GENERATION" in runbook
    assert "/workspace/var/cache/binance-vision" in runbook
    assert "never deletes or overwrites an existing generation" in normalized
    assert "reuses the shared cache" in normalized
