"""Maintained CUDA training smoke execution and evidence validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from trade_rl.operations._gpu_training_smoke_impl import (
    _load_torch_runtime,
    _load_training_performance,
    build_parser,
    build_smoke_config,
    main,
    run_gpu_training_smoke,
)

GPU_TRAINING_SMOKE_SCHEMA = "gpu_sequence_target_oracle_bc_training_smoke_v8"


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{field} must be positive")
    return resolved


def _validate_cuda_runtime(
    value: object,
    *,
    field: str,
    expected_runtime_profile: str,
) -> dict[str, object]:
    runtime = dict(_mapping(value, field=field))
    expected_mode = (
        "performance" if expected_runtime_profile == "accelerated" else "deterministic"
    )
    if runtime.get("mode") != expected_mode:
        raise ValueError(f"{field} mode does not match runtime profile")
    expected_flags: dict[str, object] = (
        {
            "deterministic_algorithms": False,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "cudnn_tf32": True,
            "float32_matmul_precision": "high",
            "matmul_tf32": True,
        }
        if expected_mode == "performance"
        else {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_tf32": False,
            "float32_matmul_precision": "highest",
            "matmul_tf32": False,
        }
    )
    for key, expected in expected_flags.items():
        if runtime.get(key) != expected:
            raise ValueError(f"{field} {key} does not match {expected_mode} mode")
    if runtime.get("sequence_encoder_autocast") not in {
        "bfloat16",
        "disabled",
    }:
        raise ValueError(f"{field} sequence_encoder_autocast is invalid")
    return runtime


def _validate_training_performance(value: object, *, field: str) -> None:
    performance = _mapping(value, field=field)
    _positive_number(
        performance.get("peak_gpu_memory_mib"),
        field=f"{field}.peak_gpu_memory_mib",
    )
    _positive_number(
        performance.get("throughput_steps_per_second"),
        field=f"{field}.throughput_steps_per_second",
    )
    artifact = _mapping(
        performance.get("training_artifact"),
        field=f"{field}.training_artifact",
    )
    if artifact.get("schema_version") != "training_performance_evidence_v1":
        raise ValueError(f"{field}.training_artifact schema is unsupported")
    if artifact.get("device_type") != "cuda":
        raise ValueError(f"{field}.training_artifact must report CUDA")
    for key in (
        "observed_environment_steps",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    ):
        _positive_integer(artifact.get(key), field=f"{field}.training_artifact.{key}")


def validate_gpu_training_smoke_evidence(
    payload: object,
    *,
    expected_commit: str,
    expected_runtime_profile: str,
    minimum_timesteps: int,
) -> dict[str, object]:
    """Validate one complete GPU smoke and resume evidence document."""

    if expected_runtime_profile not in {"compatibility", "accelerated"}:
        raise ValueError("expected_runtime_profile is unsupported")
    minimum = _positive_integer(minimum_timesteps, field="minimum_timesteps")
    evidence = dict(_mapping(payload, field="GPU smoke evidence"))
    if evidence.get("schema") != GPU_TRAINING_SMOKE_SCHEMA:
        raise ValueError("GPU smoke evidence schema is unsupported")
    if evidence.get("git_commit") != expected_commit:
        raise ValueError("GPU smoke evidence commit does not match")
    if evidence.get("runtime_profile") != expected_runtime_profile:
        raise ValueError("GPU smoke runtime profile does not match")
    if evidence.get("resolved_device") != "cuda":
        raise ValueError("GPU smoke did not resolve CUDA")
    requested = _positive_integer(
        evidence.get("requested_timesteps"),
        field="requested_timesteps",
    )
    actual = _positive_integer(evidence.get("actual_timesteps"), field="actual_timesteps")
    if requested < minimum or actual < minimum or actual < requested:
        raise ValueError("GPU smoke did not satisfy the requested timesteps")
    if evidence.get("behavior_cloning_epochs") != 1:
        raise ValueError("GPU smoke must execute one behavior cloning epoch")
    serving = _mapping(evidence.get("serving_support"), field="serving_support")
    if serving.get("status") != "supported":
        raise ValueError("GPU smoke serving support is not available")
    runtime = _validate_cuda_runtime(
        evidence.get("cuda_runtime"),
        field="CUDA runtime",
        expected_runtime_profile=expected_runtime_profile,
    )
    _validate_training_performance(evidence.get("performance"), field="performance")

    resume = _mapping(evidence.get("resume"), field="resume")
    if _positive_integer(
        resume.get("actual_timesteps"), field="resume.actual_timesteps"
    ) != actual:
        raise ValueError("resume actual timesteps do not match the initial run")
    resume_evidence = _mapping(resume.get("evidence"), field="resume.evidence")
    if resume_evidence.get("schema_version") != "training_resume_v1":
        raise ValueError("resume evidence schema is unsupported")
    resume_runtime = _validate_cuda_runtime(
        resume.get("cuda_runtime"),
        field="resume CUDA runtime",
        expected_runtime_profile=expected_runtime_profile,
    )
    if resume_runtime != runtime:
        raise ValueError("resume CUDA runtime differs from the initial run")
    _validate_training_performance(
        resume.get("performance"),
        field="resume.performance",
    )
    return evidence


def validate_gpu_training_smoke_file(
    path: Path,
    *,
    expected_commit: str,
    expected_runtime_profile: str,
    minimum_timesteps: int,
) -> dict[str, object]:
    """Load and validate one GPU smoke evidence JSON file."""

    import json

    return validate_gpu_training_smoke_evidence(
        json.loads(path.read_text(encoding="utf-8")),
        expected_commit=expected_commit,
        expected_runtime_profile=expected_runtime_profile,
        minimum_timesteps=minimum_timesteps,
    )


__all__ = [
    "GPU_TRAINING_SMOKE_SCHEMA",
    "_load_torch_runtime",
    "_load_training_performance",
    "build_parser",
    "build_smoke_config",
    "main",
    "run_gpu_training_smoke",
    "validate_gpu_training_smoke_evidence",
    "validate_gpu_training_smoke_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
