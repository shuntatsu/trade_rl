from __future__ import annotations

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
    assert "--work-root /workspace/var/binance-multitimeframe-full" in compose


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
        assert f"{name}: ${{{name}:?" in compose
    assert "^[0-9a-f]{40}$" in dockerfile
    assert '"true"|"false"' in dockerfile
