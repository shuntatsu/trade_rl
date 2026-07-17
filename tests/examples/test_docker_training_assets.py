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

    for entry in (
        ".git",
        ".venv",
        ".worktrees",
        "var",
        "data",
        "data_past",
        "output",
        "__pycache__",
    ):
        assert entry in ignored


def test_optional_training_commands_use_installed_python_as_non_root() -> None:
    runbook = (ROOT / "docs/operations/docker-gpu-full-training.md").read_text(
        encoding="utf-8"
    )

    assert "--entrypoint uv trainer run python" not in runbook
    assert (
        "--entrypoint python trainer "
        "examples/binance-multitimeframe/training_cuda_preflight.py"
    ) in runbook
    assert (
        "--entrypoint python trainer "
        "examples/binance-multitimeframe/run_gpu_training_smoke.py"
    ) in runbook


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


def test_training_dockerfile_isolates_fast_provenance_validation_stage() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")

    provenance_stage = dockerfile.index("AS provenance-validation")
    runtime_stage = dockerfile.index("AS training-runtime")
    marker_copy = dockerfile.index("COPY --from=provenance-validation")

    assert provenance_stage < runtime_stage < marker_copy
    assert dockerfile.count("ARG TRADE_RL_GIT_COMMIT") == 2
    assert dockerfile.count("ARG TRADE_RL_GIT_DIRTY") == 2


def test_training_dockerfile_keeps_heavy_dependencies_out_of_late_layers() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")
    runtime = dockerfile.split("FROM python:3.12-slim AS training-runtime", 1)[1]

    dependency_sync = runtime.index(
        "uv sync --frozen --extra train-sb3 --no-dev --no-install-project"
    )
    source_copy = runtime.index("COPY --chown=trainer:trainer trade_rl ./trade_rl")
    project_sync = runtime.index(
        "RUN uv sync --frozen --extra train-sb3 --no-dev", source_copy
    )
    commit_argument = runtime.index("ARG TRADE_RL_GIT_COMMIT")
    marker_copy = runtime.index("COPY --from=provenance-validation")

    assert dependency_sync < source_copy < project_sync < commit_argument < marker_copy
    assert "chown -R" not in dockerfile
    assert "chown trainer:trainer /workspace/var" in runtime
    assert "chown" not in runtime[commit_argument:]


def test_training_image_build_checks_non_root_runtime_contract() -> None:
    dockerfile = (ROOT / "Dockerfile.training").read_text(encoding="utf-8")
    user = dockerfile.index("USER trainer")
    runtime_contract = dockerfile.index('test "$(id -u)" -ne 0', user)

    assert "test -w /workspace/var" in dockerfile[runtime_contract:]
    assert "test -r /workspace/trade_rl/__init__.py" in dockerfile[runtime_contract:]
    assert "test -r /workspace/examples" in dockerfile[runtime_contract:]
    assert "cat /provenance.valid" in dockerfile[runtime_contract:]


def test_provenance_validation_target_fails_fast_without_valid_arguments() -> None:
    base_command = [
        "docker",
        "build",
        "--progress",
        "plain",
        "--target",
        "provenance-validation",
        "-f",
        "Dockerfile.training",
    ]
    cases = (
        ([], False),
        (
            [
                "--build-arg",
                f"TRADE_RL_GIT_COMMIT={'a' * 40}",
                "--build-arg",
                "TRADE_RL_GIT_DIRTY=invalid",
            ],
            False,
        ),
        (
            [
                "--build-arg",
                f"TRADE_RL_GIT_COMMIT={'a' * 40}",
                "--build-arg",
                "TRADE_RL_GIT_DIRTY=false",
            ],
            True,
        ),
    )

    for arguments, succeeds in cases:
        completed = subprocess.run(
            [*base_command, *arguments, "."],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        assert (completed.returncode == 0) is succeeds, output
        assert "uv sync --frozen" not in output


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


def test_ci_builds_and_probes_the_complete_training_image() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "training-image:" in workflow
    assert "docker build" in workflow
    assert "--build-arg TRADE_RL_GIT_COMMIT=" in workflow
    assert "--build-arg TRADE_RL_GIT_DIRTY=false" in workflow
    assert "docker run --rm --entrypoint sh" in workflow
    assert 'test "$(id -u)" -ne 0' in workflow


def test_gpu_nightly_contract_measures_vram_throughput_and_resume() -> None:
    workflow = (ROOT / ".github" / "workflows" / "gpu-nightly.yml").read_text(
        encoding="utf-8"
    )
    smoke = (
        ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "self-hosted" in workflow
    assert "nvidia" in workflow.lower()
    assert "run_gpu_training_smoke.py" in workflow
    assert "gpu-training-smoke.json" in workflow
    assert "peak_gpu_memory_mib" in smoke
    assert "throughput_steps_per_second" in smoke
    assert "resume_checkpoint" in smoke
    assert "gpu_sequence_target_oracle_bc_training_smoke_v5" in smoke


def test_ci_explicitly_runs_recovery_and_structured_serving_smokes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "Recovery and structured serving smoke" in workflow
    assert "test_backend_resumes_ppo_checkpoint_to_requested_total" in workflow
    assert "test_structured_sb3_loader_rebuilds_native_sequence_observation" in workflow
    assert "agent/causal-sequence-feature-encoder" in workflow


def test_architecture_docs_state_current_research_and_runtime_boundaries() -> None:
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    research = (ROOT / "docs" / "MULTITIMEFRAME_RESEARCH.md").read_text(
        encoding="utf-8"
    )
    runbook = (ROOT / "docs" / "operations" / "docker-gpu-full-training.md").read_text(
        encoding="utf-8"
    )
    combined = "\n".join((architecture, research, runbook))
    for phrase in (
        "approximate portfolio teacher",
        "shared per-asset actor",
        "index-backed rollout",
        "180 OOS days",
        "fresh confirmation",
        "structured sequence serving",
        "Production remains NO-GO",
    ):
        assert phrase in combined
    assert "exact optimal Oracle" not in combined
