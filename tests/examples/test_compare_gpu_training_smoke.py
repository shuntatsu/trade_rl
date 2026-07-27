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


def _load_comparison() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_gpu_training_smoke",
        COMPARISON,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GPU comparison module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_performance(*, steps: int, wall: float, throughput: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "training_performance_evidence_v1",
        "device_type": "cuda",
        "requested_environment_steps": steps,
        "observed_environment_steps": steps,
        "wall_clock_seconds": wall,
        "environment_steps_per_second": throughput,
        "collect_rollouts_seconds": wall * 0.6,
        "optimization_seconds": wall * 0.3,
        "environment_step_seconds": wall * 0.2,
        "feature_extraction_host_seconds": wall * 0.25,
        "sequence_reconstruction_seconds": wall * 0.1,
        "sequence_tensor_conversion_seconds": wall * 0.05,
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


def _smoke_payload(
    *,
    commit: str,
    profile: str,
    throughput: float,
    duration: float,
    schema: str = "gpu_sequence_target_oracle_bc_training_smoke_v7",
) -> dict[str, object]:
    steps = 128
    payload: dict[str, object] = {
        "schema": schema,
        "git_commit": commit,
        "runtime_profile": profile,
        "actual_timesteps": steps,
        "requested_timesteps": steps,
        "n_envs": 4,
        "behavior_cloning_epochs": 1,
        "resolved_device": "cuda",
        "performance": {
            "duration_seconds": duration,
            "peak_gpu_memory_mib": 512.0,
            "throughput_steps_per_second": throughput,
            "training_artifact": _training_performance(
                steps=steps,
                wall=duration * 0.8,
                throughput=steps / (duration * 0.8),
            ),
        },
    }
    return payload


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_comparison_uses_medians_and_binds_digest(tmp_path: Path) -> None:
    baseline_commit = "a" * 40
    candidate_commit = "b" * 40
    baseline = [
        _write(
            tmp_path / f"baseline-{index}.json",
            _smoke_payload(
                commit=baseline_commit,
                profile="compatibility",
                throughput=throughput,
                duration=duration,
            ),
        )
        for index, (throughput, duration) in enumerate(
            ((64.0, 2.0), (32.0, 4.0), (42.6666666667, 3.0))
        )
    ]
    candidate = [
        _write(
            tmp_path / f"candidate-{index}.json",
            _smoke_payload(
                commit=candidate_commit,
                profile="accelerated",
                throughput=throughput,
                duration=duration,
            ),
        )
        for index, (throughput, duration) in enumerate(
            ((128.0, 1.0), (64.0, 2.0), (85.3333333333, 1.5))
        )
    ]

    evidence = _load_comparison().compare_gpu_training_smokes(
        baseline_paths=baseline,
        candidate_paths=candidate,
        baseline_ref=baseline_commit,
        candidate_ref=candidate_commit,
    )

    assert evidence["schema_version"] == "gpu_training_performance_comparison_v1"
    assert evidence["baseline"]["sample_count"] == 3
    assert evidence["candidate"]["sample_count"] == 3
    assert evidence["baseline"]["medians"]["external_duration_seconds"] == 3.0
    assert evidence["candidate"]["medians"]["external_duration_seconds"] == 1.5
    assert evidence["ratios"]["external_wall_clock_speedup"] == pytest.approx(2.0)
    assert evidence["ratios"]["external_throughput"] == pytest.approx(2.0)
    digest = evidence["digest"]
    unsigned = dict(evidence)
    unsigned.pop("digest")
    assert digest == content_digest(unsigned)


def test_comparison_rejects_mismatched_workloads(tmp_path: Path) -> None:
    baseline_payload = _smoke_payload(
        commit="a" * 40,
        profile="compatibility",
        throughput=64.0,
        duration=2.0,
    )
    candidate_payload = _smoke_payload(
        commit="b" * 40,
        profile="accelerated",
        throughput=80.0,
        duration=1.6,
    )
    candidate_payload["actual_timesteps"] = 64

    with pytest.raises(ValueError, match="workload"):
        _load_comparison().compare_gpu_training_smokes(
            baseline_paths=[_write(tmp_path / "baseline.json", baseline_payload)],
            candidate_paths=[_write(tmp_path / "candidate.json", candidate_payload)],
            baseline_ref="a" * 40,
            candidate_ref="b" * 40,
        )


def test_comparison_rejects_invalid_training_digest(tmp_path: Path) -> None:
    payload = _smoke_payload(
        commit="a" * 40,
        profile="compatibility",
        throughput=64.0,
        duration=2.0,
    )
    performance = payload["performance"]
    assert isinstance(performance, dict)
    training = performance["training_artifact"]
    assert isinstance(training, dict)
    training["digest"] = "0" * 64

    with pytest.raises(ValueError, match="digest"):
        _load_comparison().compare_gpu_training_smokes(
            baseline_paths=[_write(tmp_path / "baseline.json", payload)],
            candidate_paths=[
                _write(
                    tmp_path / "candidate.json",
                    _smoke_payload(
                        commit="b" * 40,
                        profile="accelerated",
                        throughput=80.0,
                        duration=1.6,
                    ),
                )
            ],
            baseline_ref="a" * 40,
            candidate_ref="b" * 40,
        )


def test_comparison_requires_accelerated_candidate_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="accelerated"):
        _load_comparison().compare_gpu_training_smokes(
            baseline_paths=[
                _write(
                    tmp_path / "baseline.json",
                    _smoke_payload(
                        commit="a" * 40,
                        profile="compatibility",
                        throughput=64.0,
                        duration=2.0,
                    ),
                )
            ],
            candidate_paths=[
                _write(
                    tmp_path / "candidate.json",
                    _smoke_payload(
                        commit="b" * 40,
                        profile="compatibility",
                        throughput=80.0,
                        duration=1.6,
                    ),
                )
            ],
            baseline_ref="a" * 40,
            candidate_ref="b" * 40,
        )
