#!/usr/bin/env python3
"""Compare repeated baseline and accelerated GPU training smoke evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_COMPARISON_SCHEMA = "gpu_training_performance_comparison_v1"
_TRAINING_SCHEMA = "training_performance_evidence_v1"
_SMOKE_SCHEMAS = {
    "gpu_sequence_target_oracle_bc_training_smoke_v6",
    "gpu_sequence_target_oracle_bc_training_smoke_v7",
}
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_PROFILES = {"compatibility", "accelerated"}
_METRIC_FIELDS = (
    "external_duration_seconds",
    "external_throughput_steps_per_second",
    "external_peak_gpu_memory_mib",
    "training_wall_clock_seconds",
    "training_throughput_steps_per_second",
    "collect_rollouts_seconds",
    "optimization_seconds",
    "environment_step_seconds",
    "feature_extraction_host_seconds",
    "sequence_reconstruction_seconds",
    "sequence_tensor_conversion_seconds",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
)


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite and positive")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return resolved


def _non_negative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be finite and non-negative")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return resolved


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"GPU smoke evidence is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"GPU smoke evidence is invalid: {path}") from error
    return _mapping(payload, field=str(path))


def _load_training_performance(value: object) -> dict[str, object]:
    payload = _mapping(value, field="training_artifact")
    if payload.get("schema_version") != _TRAINING_SCHEMA:
        raise ValueError("training performance schema is unsupported")
    if payload.get("device_type") != "cuda":
        raise ValueError("GPU comparison requires CUDA training evidence")
    requested = _positive_integer(
        payload.get("requested_environment_steps"),
        field="requested_environment_steps",
    )
    observed = _positive_integer(
        payload.get("observed_environment_steps"),
        field="observed_environment_steps",
    )
    wall = _positive_float(
        payload.get("wall_clock_seconds"), field="wall_clock_seconds"
    )
    throughput = _positive_float(
        payload.get("environment_steps_per_second"),
        field="environment_steps_per_second",
    )
    if not math.isclose(throughput, observed / wall, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("training performance throughput is inconsistent")
    for field in (
        "collect_rollouts_seconds",
        "optimization_seconds",
        "environment_step_seconds",
        "feature_extraction_host_seconds",
        "sequence_reconstruction_seconds",
        "sequence_tensor_conversion_seconds",
    ):
        _non_negative_float(payload.get(field), field=field)
    allocated = _positive_integer(
        payload.get("peak_cuda_allocated_bytes"),
        field="peak_cuda_allocated_bytes",
    )
    reserved = _positive_integer(
        payload.get("peak_cuda_reserved_bytes"),
        field="peak_cuda_reserved_bytes",
    )
    digest = payload.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("training performance digest is invalid")
    unsigned = dict(payload)
    unsigned.pop("digest", None)
    if digest != content_digest(unsigned):
        raise ValueError("training performance digest mismatch")
    payload["requested_environment_steps"] = requested
    payload["observed_environment_steps"] = observed
    payload["wall_clock_seconds"] = wall
    payload["environment_steps_per_second"] = throughput
    payload["peak_cuda_allocated_bytes"] = allocated
    payload["peak_cuda_reserved_bytes"] = reserved
    return payload


def _load_sample(path: Path, *, legacy_profile: str | None) -> dict[str, object]:
    payload = _load_json(path)
    schema = payload.get("schema")
    if schema not in _SMOKE_SCHEMAS:
        raise ValueError("GPU smoke schema is unsupported")
    if payload.get("resolved_device") != "cuda":
        raise ValueError("GPU comparison requires resolved CUDA evidence")
    requested = _positive_integer(
        payload.get("requested_timesteps"),
        field="requested_timesteps",
    )
    actual = _positive_integer(
        payload.get("actual_timesteps"), field="actual_timesteps"
    )
    n_envs = _positive_integer(payload.get("n_envs"), field="n_envs")
    behavior_cloning_epochs = _non_negative_integer(
        payload.get("behavior_cloning_epochs"),
        field="behavior_cloning_epochs",
    )
    performance = _mapping(payload.get("performance"), field="performance")
    training = _load_training_performance(performance.get("training_artifact"))
    if training["observed_environment_steps"] != actual:
        raise ValueError(
            "GPU comparison workload differs between training artifact and smoke"
        )
    if schema == "gpu_sequence_target_oracle_bc_training_smoke_v6":
        if legacy_profile is None:
            raise ValueError("accelerated candidate requires schema v7 evidence")
        profile = legacy_profile
        commit: str | None = None
    else:
        profile = payload.get("runtime_profile")
        if profile not in _RUNTIME_PROFILES:
            raise ValueError("GPU smoke runtime profile is unsupported")
        raw_commit = payload.get("git_commit")
        if (
            not isinstance(raw_commit, str)
            or _GIT_COMMIT_PATTERN.fullmatch(raw_commit) is None
        ):
            raise ValueError("GPU smoke git commit is invalid")
        commit = raw_commit
    metrics = {
        "external_duration_seconds": _positive_float(
            performance.get("duration_seconds"),
            field="duration_seconds",
        ),
        "external_throughput_steps_per_second": _positive_float(
            performance.get("throughput_steps_per_second"),
            field="throughput_steps_per_second",
        ),
        "external_peak_gpu_memory_mib": _positive_float(
            performance.get("peak_gpu_memory_mib"),
            field="peak_gpu_memory_mib",
        ),
        "training_wall_clock_seconds": float(training["wall_clock_seconds"]),
        "training_throughput_steps_per_second": float(
            training["environment_steps_per_second"]
        ),
        "collect_rollouts_seconds": float(training["collect_rollouts_seconds"]),
        "optimization_seconds": float(training["optimization_seconds"]),
        "environment_step_seconds": float(training["environment_step_seconds"]),
        "feature_extraction_host_seconds": float(
            training["feature_extraction_host_seconds"]
        ),
        "sequence_reconstruction_seconds": float(
            training["sequence_reconstruction_seconds"]
        ),
        "sequence_tensor_conversion_seconds": float(
            training["sequence_tensor_conversion_seconds"]
        ),
        "peak_cuda_allocated_bytes": float(training["peak_cuda_allocated_bytes"]),
        "peak_cuda_reserved_bytes": float(training["peak_cuda_reserved_bytes"]),
    }
    return {
        "path": str(path),
        "schema": schema,
        "git_commit": commit,
        "runtime_profile": profile,
        "workload": {
            "requested_timesteps": requested,
            "actual_timesteps": actual,
            "n_envs": n_envs,
            "behavior_cloning_epochs": behavior_cloning_epochs,
        },
        "metrics": metrics,
    }


def _aggregate(
    paths: list[Path],
    *,
    ref: str,
    expected_profile: str,
    legacy_profile: str | None,
) -> dict[str, object]:
    if not paths:
        raise ValueError("GPU comparison requires at least one sample per side")
    samples = [_load_sample(path, legacy_profile=legacy_profile) for path in paths]
    first_workload = samples[0]["workload"]
    for sample in samples:
        if sample["runtime_profile"] != expected_profile:
            raise ValueError(
                f"GPU comparison requires {expected_profile} runtime evidence"
            )
        if sample["workload"] != first_workload:
            raise ValueError("GPU comparison sample workload mismatch")
        commit = sample["git_commit"]
        if commit is not None and _GIT_COMMIT_PATTERN.fullmatch(ref) and commit != ref:
            raise ValueError("GPU comparison sample commit does not match its ref")
    medians = {
        field: float(median(float(sample["metrics"][field]) for sample in samples))
        for field in _METRIC_FIELDS
    }
    return {
        "ref": ref,
        "runtime_profile": expected_profile,
        "sample_count": len(samples),
        "workload": first_workload,
        "medians": medians,
        "samples": samples,
    }


def _positive_ratio(numerator: float, denominator: float, *, field: str) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError(f"{field} ratio requires positive values")
    return numerator / denominator


def compare_gpu_training_smokes(
    *,
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    baseline_ref: str,
    candidate_ref: str,
) -> dict[str, object]:
    """Validate repeated evidence and return one digest-bound median comparison."""

    for field, ref in (
        ("baseline_ref", baseline_ref),
        ("candidate_ref", candidate_ref),
    ):
        if _GIT_COMMIT_PATTERN.fullmatch(ref) is None:
            raise ValueError(f"{field} must be a lowercase 40-character commit")
    baseline = _aggregate(
        baseline_paths,
        ref=baseline_ref,
        expected_profile="compatibility",
        legacy_profile="compatibility",
    )
    candidate = _aggregate(
        candidate_paths,
        ref=candidate_ref,
        expected_profile="accelerated",
        legacy_profile=None,
    )
    if baseline["workload"] != candidate["workload"]:
        raise ValueError("GPU comparison workload mismatch")
    baseline_medians = baseline["medians"]
    candidate_medians = candidate["medians"]
    assert isinstance(baseline_medians, dict)
    assert isinstance(candidate_medians, dict)
    ratios = {
        "external_wall_clock_speedup": _positive_ratio(
            float(baseline_medians["external_duration_seconds"]),
            float(candidate_medians["external_duration_seconds"]),
            field="external_wall_clock_speedup",
        ),
        "external_throughput": _positive_ratio(
            float(candidate_medians["external_throughput_steps_per_second"]),
            float(baseline_medians["external_throughput_steps_per_second"]),
            field="external_throughput",
        ),
        "training_wall_clock_speedup": _positive_ratio(
            float(baseline_medians["training_wall_clock_seconds"]),
            float(candidate_medians["training_wall_clock_seconds"]),
            field="training_wall_clock_speedup",
        ),
        "training_throughput": _positive_ratio(
            float(candidate_medians["training_throughput_steps_per_second"]),
            float(baseline_medians["training_throughput_steps_per_second"]),
            field="training_throughput",
        ),
        "external_peak_gpu_memory": _positive_ratio(
            float(candidate_medians["external_peak_gpu_memory_mib"]),
            float(baseline_medians["external_peak_gpu_memory_mib"]),
            field="external_peak_gpu_memory",
        ),
        "peak_cuda_allocated": _positive_ratio(
            float(candidate_medians["peak_cuda_allocated_bytes"]),
            float(baseline_medians["peak_cuda_allocated_bytes"]),
            field="peak_cuda_allocated",
        ),
        "peak_cuda_reserved": _positive_ratio(
            float(candidate_medians["peak_cuda_reserved_bytes"]),
            float(baseline_medians["peak_cuda_reserved_bytes"]),
            field="peak_cuda_reserved",
        ),
    }
    unsigned: dict[str, object] = {
        "schema_version": _COMPARISON_SCHEMA,
        "workload": baseline["workload"],
        "baseline": baseline,
        "candidate": candidate,
        "ratios": ratios,
        "acceptance": {
            "speedup_threshold_enforced": False,
            "production_status": "NO-GO",
        },
    }
    return {**unsigned, "digest": content_digest(unsigned)}


def write_gpu_performance_comparison(path: Path, evidence: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(evidence) + b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/gpu-performance-comparison.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence = compare_gpu_training_smokes(
        baseline_paths=args.baseline,
        candidate_paths=args.candidate,
        baseline_ref=args.baseline_ref,
        candidate_ref=args.candidate_ref,
    )
    write_gpu_performance_comparison(args.output, evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
