from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from trade_rl.artifacts.hashing import content_digest

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = (
    ROOT
    / "examples"
    / "binance-multitimeframe"
    / "compare_gpu_training_smoke.py"
)
WORKFLOW = ROOT / ".github" / "workflows" / "gpu-performance-comparison.yml"
BASELINE_REF = "1f597caf85fe5200fe7abc34461236b65ebb8b1d"


def _load_comparison() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_gpu_training_smoke_identity",
        COMPARISON,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GPU comparison module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_performance() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "training_performance_evidence_v1",
        "device_type": "cuda",
        "requested_environment_steps": 128,
        "observed_environment_steps": 128,
        "wall_clock_seconds": 2.0,
        "environment_steps_per_second": 64.0,
        "collect_rollouts_seconds": 1.2,
        "optimization_seconds": 0.6,
        "environment_step_seconds": 0.4,
        "feature_extraction_host_seconds": 0.5,
        "sequence_reconstruction_seconds": 0.2,
        "sequence_tensor_conversion_seconds": 0.1,
        "collect_rollouts_calls": 2,
        "optimization_calls": 2,
        "environment_step_calls": 16,
        "feature_extraction_calls": 24,
        "sequence_reconstruction_calls": 2,
        "sequence_tensor_conversion_calls": 2,
        "peak_cuda_allocated_bytes": 1_024,
        "peak_cuda_reserved_bytes": 2_048,
        "component_timers_overlap": True,
    }
    payload["digest"] = content_digest(payload)
    return payload


def _smoke(
    *,
    schema: str,
    profile: str | None = None,
    commit: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": schema,
        "actual_timesteps": 128,
        "requested_timesteps": 128,
        "n_envs": 4,
        "behavior_cloning_epochs": 1,
        "resolved_device": "cuda",
        "performance": {
            "duration_seconds": 2.0,
            "peak_gpu_memory_mib": 512.0,
            "throughput_steps_per_second": 64.0,
            "training_artifact": _training_performance(),
        },
    }
    if profile is not None:
        payload["runtime_profile"] = profile
    if commit is not None:
        payload["git_commit"] = commit
    return payload


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_accelerated_candidate_rejects_legacy_v6_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema v7"):
        _load_comparison().compare_gpu_training_smokes(
            baseline_paths=[
                _write(
                    tmp_path / "baseline.json",
                    _smoke(schema="gpu_sequence_target_oracle_bc_training_smoke_v6"),
                )
            ],
            candidate_paths=[
                _write(
                    tmp_path / "candidate.json",
                    _smoke(schema="gpu_sequence_target_oracle_bc_training_smoke_v6"),
                )
            ],
            baseline_ref=BASELINE_REF,
            candidate_ref="b" * 40,
        )


def test_v7_candidate_requires_explicit_commit_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="git commit"):
        _load_comparison().compare_gpu_training_smokes(
            baseline_paths=[
                _write(
                    tmp_path / "baseline.json",
                    _smoke(schema="gpu_sequence_target_oracle_bc_training_smoke_v6"),
                )
            ],
            candidate_paths=[
                _write(
                    tmp_path / "candidate.json",
                    _smoke(
                        schema="gpu_sequence_target_oracle_bc_training_smoke_v7",
                        profile="accelerated",
                    ),
                )
            ],
            baseline_ref=BASELINE_REF,
            candidate_ref="b" * 40,
        )


def test_workflow_fixes_the_verified_h1_baseline() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert BASELINE_REF in workflow
    assert "${{ inputs.baseline_ref }}" not in workflow
    assert f'REQUESTED_BASELINE_REF: {BASELINE_REF}' in workflow
