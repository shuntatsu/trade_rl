#!/usr/bin/env python3
"""Run one tiny deterministic training job through the authoritative workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from trade_rl._source_checkout import source_checkout_root
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import MarketDataset, write_market_dataset_files
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.rl.training import gamma_from_half_life
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = source_checkout_root()
_TEMPLATE = ROOT / "examples" / "quickstart" / "training.json"
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_template() -> dict[str, Any]:
    payload = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quickstart training template must be a JSON object")
    return dict(payload)


def _packaged_git_provenance() -> tuple[str, bool]:
    commit = os.environ.get("TRADE_RL_GIT_COMMIT", "")
    if not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(
            "TRADE_RL_GIT_COMMIT must be a 40-character lowercase Git commit"
        )
    dirty = os.environ.get("TRADE_RL_GIT_DIRTY")
    if dirty not in {"true", "false"}:
        raise ValueError("TRADE_RL_GIT_DIRTY must be exactly true or false")
    return commit, dirty == "true"


def _smoke_config_payload(
    timesteps: int, runtime_profile: str = "compatibility"
) -> dict[str, Any]:
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:
        raise ValueError("timesteps must be a positive integer")
    if runtime_profile not in {"compatibility", "accelerated"}:
        raise ValueError("runtime_profile must be compatibility or accelerated")
    accelerated = runtime_profile == "accelerated"
    payload = _load_template()
    git_commit, git_dirty = _packaged_git_provenance()
    payload["git_commit"] = git_commit
    payload["git_dirty"] = git_dirty
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("quickstart training template has no training object")
    training.update(
        {
            "observation_encoder": "hierarchical_sequence_v2",
            "policy_actor_head": "hierarchical_gate_target_v1",
            "asset_embedding_dim": 64,
            "global_embedding_dim": 64,
            "batch_size": 32,
            "device": "cuda",
            "cuda_runtime_mode": ("performance" if accelerated else "deterministic"),
            "n_envs": 4,
            "n_steps": 8,
            "n_epochs": 3,
            "policy": "MultiInputPolicy",
            "policy_net_arch": [128, 64],
            "value_net_arch": [192, 96],
            "sequence_tcn_capacity": "compact",
            "sequence_d_model": 128,
            "sequence_timeframe_attention_heads": 4,
            "sequence_asset_attention_heads": 4,
            "sequence_timeframe_attention_layers": 1,
            "sequence_asset_attention_layers": 1,
            "sequence_dropout": 0.05,
            "sequence_compile": accelerated,
            "sequence_compile_mode": "reduce-overhead",
            "sequence_transfer_mode": (
                "pinned_non_blocking" if accelerated else "synchronous"
            ),
            "vector_environment_mode": ("subprocess" if accelerated else "in_process"),
            "max_policy_parameters": 2_500_000,
            "max_rollout_buffer_bytes": 268_435_456,
            "checkpoint_interval_steps": max(1, timesteps // 2),
            "max_checkpoints": 2,
            "seeds": [0],
            "timesteps": timesteps,
            "behavior_cloning_epochs": 1,
            "behavior_cloning_batch_size": 32,
            "behavior_cloning_validation_fraction": 0.1,
            "decision_hours": 0.25,
            "discount_half_life_hours": 168.0,
            "gamma": gamma_from_half_life(decision_hours=0.25, half_life_hours=168.0),
        }
    )
    payload["environment"] = {
        "episode_hours": 1.0,
        "decision_hours": 0.25,
        "initial_capital": 100_000.0,
        "finite_horizon_observation": True,
        "initial_state_modes": ["cash"],
        "structured_sequence_observation": True,
        "require_full_reward_preroll": True,
        "sequence_windows": [["15m", 96], ["1h", 168], ["4h", 120], ["1d", 60]],
    }
    payload["trend"] = {
        "fast_hours": 4.0,
        "base_hours": 12.0,
        "slow_hours": 24.0,
        "mode": "time_series",
        "signal_scale": 0.05,
    }
    payload["risk"]["max_turnover"] = None
    payload["action"] = {
        "mode": "target_weight",
        "alpha_enabled": False,
        "risk_tilt_enabled": False,
        "n_factors": 0,
        "target_weight_count": 1,
        "validation_mode": "clip",
    }
    return payload


def _build_sequence_smoke_dataset(n_bars: int = 5_680) -> MarketDataset:
    if n_bars < 5_680:
        raise ValueError("sequence smoke dataset needs at least 5680 bars")
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    phase = np.arange(n_bars, dtype=np.float64)
    returns = 0.00005 + 0.0004 * np.sin(phase / 47.0)
    close = 30_000.0 * np.exp(np.cumsum(returns))
    open_price = np.concatenate(([close[0]], close[:-1]))
    spread = 0.001 + 0.0002 * np.cos(phase / 19.0)
    features = np.stack(
        tuple(
            np.sin(phase / float(11 + index % 97))
            + 0.1 * np.cos(phase / float(7 + index % 43))
            for index in range(len(specs))
        ),
        axis=1,
    ).astype(np.float32)[:, None, :]
    timestamps = np.datetime64("2025-01-01T00:15:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(15, "m")
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=features,
        global_features=np.stack(
            (np.sin(phase / 97.0), np.cos(phase / 193.0)), axis=1
        ).astype(np.float32),
        open=open_price[:, None],
        high=(np.maximum(open_price, close) * (1.0 + spread))[:, None],
        low=(np.minimum(open_price, close) * (1.0 - spread))[:, None],
        close=close[:, None],
        volume=(1_000.0 + 100.0 * np.sin(phase / 13.0))[:, None],
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones(features.shape, dtype=np.bool_),
        feature_names=tuple(spec.name for spec in specs),
        global_feature_names=("market_cycle", "risk_cycle"),
        periods_per_year=35_040,
        fee_rate=np.full((n_bars, 1), 0.0005, dtype=np.float64),
        spread_rate=np.full((n_bars, 1), 0.0002, dtype=np.float64),
        max_participation_rate=np.full((n_bars, 1), 0.05, dtype=np.float64),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
    )
    return dataset.with_content_identity({"source": "sequence-gpu-smoke-v1"})


def build_smoke_config(
    timesteps: int, runtime_profile: str = "compatibility"
) -> TrainingRunConfig:
    """Build the tiny run while retaining maintained CUDA/model dimensions."""

    return TrainingRunConfig.from_mapping(
        _smoke_config_payload(timesteps, runtime_profile=runtime_profile)
    )


def _gpu_memory_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return 0
    values: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            values.append(int(stripped))
    return sum(values)


def _run_authoritative_training(
    *, config: Path, dataset: Path, artifacts: Path, run_id: str
) -> tuple[dict[str, Any], dict[str, float]]:
    command = [
        sys.executable,
        "-m",
        "trade_rl.cli.app",
        "train",
        "run",
        "--config",
        str(config),
        "--dataset",
        str(dataset),
        "--output",
        str(artifacts),
        "--run-id",
        run_id,
    ]
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    peak_gpu_memory_mib = 0
    while process.poll() is None:
        peak_gpu_memory_mib = max(peak_gpu_memory_mib, _gpu_memory_mib())
        time.sleep(0.2)
    stdout, stderr = process.communicate()
    duration_seconds = max(time.perf_counter() - started, 1e-9)
    peak_gpu_memory_mib = max(peak_gpu_memory_mib, _gpu_memory_mib())
    if process.returncode != 0:
        raise RuntimeError("authoritative training workflow failed: " + stderr.strip())
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("authoritative training workflow returned no JSON")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("authoritative training result must be a JSON object")
    resolved = dict(result)
    artifact_path = Path(str(resolved["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    ensemble_payload = json.loads(
        (artifact_path / "ensemble.json").read_text(encoding="utf-8")
    )
    actual_timesteps = int(ensemble_payload["actual_timesteps"])
    resolved["actual_timesteps"] = actual_timesteps
    return resolved, {
        "duration_seconds": duration_seconds,
        "peak_gpu_memory_mib": float(peak_gpu_memory_mib),
        "throughput_steps_per_second": actual_timesteps / duration_seconds,
    }


def _load_training_performance(member_root: Path) -> dict[str, object]:
    path = member_root / "training-performance.json"
    if not path.is_file():
        raise RuntimeError("training performance evidence is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training performance evidence must be a JSON object")
    if payload.get("schema_version") != "training_performance_evidence_v1":
        raise ValueError("training performance schema is unsupported")
    observed = payload.get("observed_environment_steps")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        raise ValueError("training performance observed steps must be positive")
    wall = payload.get("wall_clock_seconds")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, int | float)
        or float(wall) <= 0.0
    ):
        raise ValueError("training performance wall time must be positive")
    digest = payload.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("training performance digest is invalid")
    unsigned = dict(payload)
    unsigned.pop("digest")
    if digest != content_digest(unsigned):
        raise ValueError("training performance digest mismatch")
    return dict(payload)


_TORCH_RUNTIME_FIELDS = {
    "mode",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cudnn_tf32",
    "float32_matmul_precision",
    "matmul_tf32",
    "sequence_encoder_autocast",
}


def _load_torch_runtime(
    member_root: Path,
    *,
    expected_mode: str,
) -> dict[str, object]:
    path = member_root / "model-architecture.json"
    if not path.is_file():
        raise RuntimeError("model architecture evidence is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model architecture evidence must be a JSON object")
    if payload.get("schema_version") != "policy_architecture_v2":
        raise ValueError("model architecture schema is unsupported")
    raw_runtime = payload.get("torch_runtime")
    if not isinstance(raw_runtime, dict):
        raise ValueError("torch runtime evidence must be a mapping")
    if set(raw_runtime) != _TORCH_RUNTIME_FIELDS:
        raise ValueError("torch runtime evidence fields are invalid")
    if expected_mode not in {"deterministic", "performance"}:
        raise ValueError("expected CUDA runtime mode is invalid")
    if raw_runtime.get("mode") != expected_mode:
        raise ValueError("torch runtime mode does not match the requested mode")
    for field in (
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_tf32",
        "matmul_tf32",
    ):
        if not isinstance(raw_runtime.get(field), bool):
            raise ValueError(f"torch runtime {field} must be a boolean")
    expected_flags = (
        {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_tf32": False,
            "float32_matmul_precision": "highest",
            "matmul_tf32": False,
        }
        if expected_mode == "deterministic"
        else {
            "deterministic_algorithms": False,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "cudnn_tf32": True,
            "float32_matmul_precision": "high",
            "matmul_tf32": True,
        }
    )
    for field, expected in expected_flags.items():
        if raw_runtime.get(field) != expected:
            raise ValueError(
                f"torch runtime {field} does not match {expected_mode} mode"
            )
    if raw_runtime.get("sequence_encoder_autocast") not in {
        "bfloat16",
        "disabled",
    }:
        raise ValueError("torch runtime sequence encoder autocast is invalid")
    return dict(raw_runtime)


def run_gpu_training_smoke(
    *, work_root: Path, timesteps: int, runtime_profile: str = "compatibility"
) -> dict[str, object]:
    """Preflight CUDA, train one seed, and persist inspectable smoke evidence."""

    import torch

    preflight_module = _load_module(
        "training_cuda_preflight",
        Path(__file__).with_name("training_cuda_preflight.py"),
    )
    write_cuda_preflight_evidence: Callable[[Path, Any], dict[str, object]] = getattr(
        preflight_module, "write_cuda_preflight_evidence"
    )
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    preflight = write_cuda_preflight_evidence(work_root / "cuda-preflight.json", torch)

    dataset_path = work_root / "dataset"
    write_market_dataset_files(dataset_path, _build_sequence_smoke_dataset())
    config_payload = _smoke_config_payload(timesteps, runtime_profile=runtime_profile)
    config_path = work_root / "training.json"
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = TrainingRunConfig.from_json(config_path)
    result, first_metrics = _run_authoritative_training(
        config=config_path,
        dataset=dataset_path,
        artifacts=work_root / "artifacts",
        run_id="gpu-training-smoke",
    )

    artifact_path = Path(str(result["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    ensemble = json.loads((artifact_path / "ensemble.json").read_text(encoding="utf-8"))
    serving_support = json.loads(
        (artifact_path / "serving-support.json").read_text(encoding="utf-8")
    )
    if serving_support.get("status") != "supported":
        raise RuntimeError("structured smoke must publish native serving support")
    member_root = artifact_path / "members" / "member-000"
    training_performance = _load_training_performance(member_root)
    expected_cuda_mode = str(config.training.cuda_runtime_mode)
    torch_runtime = _load_torch_runtime(member_root, expected_mode=expected_cuda_mode)
    policy = member_root / "policy.zip"
    checkpoint_manifests = sorted(
        (member_root / "checkpoints").glob("step-*/checkpoint.json")
    )
    if not checkpoint_manifests:
        raise RuntimeError("GPU smoke did not publish a resumable checkpoint")
    resume_checkpoint = checkpoint_manifests[0].parent
    resume_payload = deepcopy(config_payload)
    resume_payload["resume_checkpoints"] = {"0": str(resume_checkpoint)}
    resume_config_path = work_root / "training-resume.json"
    resume_config_path.write_text(
        json.dumps(resume_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resumed, resume_metrics = _run_authoritative_training(
        config=resume_config_path,
        dataset=dataset_path,
        artifacts=work_root / "artifacts-resumed",
        run_id="gpu-training-smoke-resumed",
    )
    resumed_artifact = Path(str(resumed["artifact_path"]))
    if not resumed_artifact.is_absolute():
        resumed_artifact = ROOT / resumed_artifact
    resumed_member_root = resumed_artifact / "members" / "member-000"
    resumed_training_performance = _load_training_performance(resumed_member_root)
    resumed_torch_runtime = _load_torch_runtime(
        resumed_member_root, expected_mode=expected_cuda_mode
    )
    resume_evidence_path = resumed_member_root / "resume.json"
    if not resume_evidence_path.is_file():
        raise RuntimeError("GPU smoke resume evidence is missing")
    evidence: dict[str, object] = {
        "actual_timesteps": int(ensemble["actual_timesteps"]),
        "git_commit": config.git_commit,
        "runtime_profile": runtime_profile,
        "checkpoint": {
            "digest": ensemble["members"][0]["checkpoint_digest"],
            "path": str(policy),
            "size_bytes": policy.stat().st_size,
        },
        "cuda_preflight": preflight,
        "n_envs": config.training.n_envs,
        "behavior_cloning_epochs": config.training.behavior_cloning_epochs,
        "cuda_runtime": torch_runtime,
        "serving_support": serving_support,
        "requested_timesteps": config.training.timesteps,
        "resolved_device": ensemble["resolved_device"],
        "performance": {
            **first_metrics,
            "training_artifact": training_performance,
        },
        "resume": {
            "actual_timesteps": int(resumed["actual_timesteps"]),
            "checkpoint": str(resume_checkpoint),
            "cuda_runtime": resumed_torch_runtime,
            "evidence": json.loads(resume_evidence_path.read_text(encoding="utf-8")),
            "performance": {
                **resume_metrics,
                "training_artifact": resumed_training_performance,
            },
        },
        "schema": "gpu_sequence_target_oracle_bc_training_smoke_v8",
    }
    evidence_path = work_root / "gpu-training-smoke.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("var/gpu-training-smoke"),
    )
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument(
        "--runtime-profile",
        choices=("compatibility", "accelerated"),
        default="compatibility",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    work_root = (
        args.work_root if args.work_root.is_absolute() else ROOT / args.work_root
    )
    evidence = run_gpu_training_smoke(
        work_root=work_root,
        timesteps=args.timesteps,
        runtime_profile=args.runtime_profile,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
