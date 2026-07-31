from __future__ import annotations

from copy import deepcopy

import pytest

from trade_rl.operations.gpu_training_smoke import (
    GPU_TRAINING_SMOKE_SCHEMA,
    validate_gpu_training_smoke_evidence,
)


def _payload() -> dict[str, object]:
    runtime = {
        "mode": "performance",
        "deterministic_algorithms": False,
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "cudnn_tf32": True,
        "float32_matmul_precision": "high",
        "matmul_tf32": True,
        "sequence_encoder_autocast": "bfloat16",
    }
    performance = {
        "peak_gpu_memory_mib": 128.0,
        "throughput_steps_per_second": 32.0,
        "training_artifact": {
            "schema_version": "training_performance_evidence_v1",
            "device_type": "cuda",
            "observed_environment_steps": 128,
            "peak_cuda_allocated_bytes": 1024,
            "peak_cuda_reserved_bytes": 2048,
        },
    }
    return {
        "schema": GPU_TRAINING_SMOKE_SCHEMA,
        "git_commit": "a" * 40,
        "runtime_profile": "accelerated",
        "resolved_device": "cuda",
        "requested_timesteps": 128,
        "actual_timesteps": 128,
        "behavior_cloning_epochs": 1,
        "cuda_runtime": runtime,
        "serving_support": {"status": "supported"},
        "performance": performance,
        "resume": {
            "actual_timesteps": 128,
            "cuda_runtime": runtime,
            "evidence": {"schema_version": "training_resume_v1"},
            "performance": performance,
        },
    }


def test_gpu_training_smoke_schema_is_v8() -> None:
    assert (
        GPU_TRAINING_SMOKE_SCHEMA == "gpu_sequence_target_oracle_bc_training_smoke_v8"
    )


def test_validator_accepts_complete_accelerated_cuda_evidence() -> None:
    validated = validate_gpu_training_smoke_evidence(
        _payload(),
        expected_commit="a" * 40,
        expected_runtime_profile="accelerated",
        minimum_timesteps=128,
    )

    assert validated["schema"] == GPU_TRAINING_SMOKE_SCHEMA
    assert validated["resolved_device"] == "cuda"


def test_validator_rejects_previous_schema() -> None:
    payload = _payload()
    payload["schema"] = "gpu_sequence_target_oracle_bc_training_smoke_v7"

    with pytest.raises(ValueError, match="schema"):
        validate_gpu_training_smoke_evidence(
            payload,
            expected_commit="a" * 40,
            expected_runtime_profile="accelerated",
            minimum_timesteps=128,
        )


def test_validator_rejects_resume_runtime_drift() -> None:
    payload = _payload()
    resume = deepcopy(payload["resume"])
    assert isinstance(resume, dict)
    resume_runtime = deepcopy(resume["cuda_runtime"])
    assert isinstance(resume_runtime, dict)
    resume_runtime["matmul_tf32"] = False
    resume["cuda_runtime"] = resume_runtime
    payload["resume"] = resume

    with pytest.raises(ValueError, match="resume CUDA runtime"):
        validate_gpu_training_smoke_evidence(
            payload,
            expected_commit="a" * 40,
            expected_runtime_profile="accelerated",
            minimum_timesteps=128,
        )
